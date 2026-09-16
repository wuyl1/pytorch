from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial

import torch


AllGatherCopyIn = Callable[
    [list[torch.Tensor], torch.Tensor, list[int], int, int],
    tuple[torch.Tensor, torch.Tensor],
]


class AllGatherLayout(ABC):
    """Optional input packing and output layout for an all-gather backend."""

    def prepare(
        self,
        input_split_sizes: list[int],
        input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
        param_input_dtypes: list[list[torch.dtype]],
        param_input_numels: list[list[int]],
        can_use_param_contiguous_output: bool,
        owner_token: int,
    ) -> tuple[AllGatherCopyIn, object | None]:
        """Select input packing and metadata before allocating the output."""
        metadata = self.prepare_output(
            input_split_sizes,
            input_numel,
            world_size,
            dtype,
            device,
            param_input_dtypes,
            param_input_numels,
            can_use_param_contiguous_output,
            owner_token,
        )
        if metadata is None:
            return torch.ops.fsdp.all_gather_copy_in, None
        return partial(self.copy_in, output_metadata=metadata), metadata

    @abstractmethod
    def prepare_output(
        self,
        input_split_sizes: list[int],
        input_numel: int,
        world_size: int,
        dtype: torch.dtype,
        device: torch.device,
        param_input_dtypes: list[list[torch.dtype]],
        param_input_numels: list[list[int]],
        can_use_param_contiguous_output: bool,
        owner_token: int,
    ) -> object | None:
        """Return per-call metadata, or None to use rank-major input and output.

        The backend must produce the selected layout for this collective.
        Metadata must remain valid until its result is finalized.
        """
        ...

    def copy_in(
        self,
        all_gather_inputs: list[torch.Tensor],
        all_gather_output: torch.Tensor,
        all_gather_input_split_sizes: list[int],
        all_gather_input_numel: int,
        rank: int,
        output_metadata: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack inputs for a collective using this layout."""
        return torch.ops.fsdp.all_gather_copy_in(
            all_gather_inputs,
            all_gather_output,
            all_gather_input_split_sizes,
            all_gather_input_numel,
            rank,
        )

    @abstractmethod
    def finalize_outputs(
        self,
        all_gather_output: torch.Tensor,
        param_input_numels: list[list[int]],
        world_size: int,
        output_metadata: object,
    ) -> list[list[torch.Tensor]]:
        """Return the per-parameter outputs after collective completion."""
        ...

    def param_contiguous_output_views(
        self,
        all_gather_output: torch.Tensor,
        param_input_numels: list[list[int]],
        world_size: int,
    ) -> list[list[torch.Tensor]]:
        """Carve a parameter-contiguous output into per-parameter views."""
        output_offset = 0
        outputs: list[list[torch.Tensor]] = []
        for input_numels in param_input_numels:
            output_numel = input_numels[0] * world_size
            param_output = all_gather_output.narrow(0, output_offset, output_numel)
            outputs.append([param_output])
            output_offset += output_numel
        if output_offset != all_gather_output.numel():
            raise AssertionError(
                "parameter-contiguous all-gather output covered "
                f"{output_offset} of {all_gather_output.numel()} elements"
            )
        return outputs
