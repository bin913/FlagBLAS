from backend_utils import VendorInfoBase

vendor_info = VendorInfoBase(
    vendor_name="hygon",
    device_name="cuda",
    device_query_cmd="hy-smi",
    triton_extra_name="hip",
)

CUSTOMIZED_UNUSED_OPS = ()

__all__ = ["*"]
