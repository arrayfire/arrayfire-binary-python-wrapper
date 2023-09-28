import ctypes

from arrayfire_wrapper.defines import AFArray, CDimT
from arrayfire_wrapper.lib._constants import Storage
from arrayfire_wrapper.lib._utility import call_from_clib


def sparse_to_dense(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__dense.htm#ga80c3d8db78d537b74d9caebcf359b6a5
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sparse_to_dense.__name__, ctypes.pointer(out), arr)
    return out


def create_sparse_array(
    n_rows: int, n_cols: int, values: AFArray, row_idx: AFArray, col_idx: AFArray, storage_type: Storage, /
) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__create.htm#ga42c5cf729a232c1cbbcfe0f664f3b986
    """
    out = AFArray.create_null_pointer()
    call_from_clib(
        create_sparse_array.__name__,
        ctypes.pointer(out),
        CDimT(n_rows),
        CDimT(n_cols),
        values,
        row_idx,
        col_idx,
        storage_type.value,
    )
    return out


def create_sparse_array_from_dense(arr: AFArray, storage_type: Storage, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__create.htm#ga52e3b2895cf9e9d697a06b4b44190d92
    """
    out = AFArray.create_null_pointer()
    call_from_clib(create_sparse_array_from_dense.__name__, ctypes.pointer(out), arr, storage_type.value)
    return out


def create_sparse_array_from_ptr() -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__create.htm#ga9a0ae91eea18203041d9f9131dbb99cc
    """
    # TODO
    return NotImplemented


def sparse_convert_to(arr: AFArray, storage_type: Storage, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__convert__to.htm#ga64556aa9252f8af116a268599cc66f68
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sparse_convert_to.__name__, ctypes.pointer(out), arr, storage_type.value)
    return out


def sparse_get_col_idx(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__col__idx.htm#gaa62f2eaac514763e871e96a878155fb2
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sparse_get_col_idx.__name__, ctypes.pointer(out), arr)
    return out


def sparse_get_info(arr: AFArray, /) -> tuple[AFArray, AFArray, AFArray, Storage]:
    """
    source: https://arrayfire.org/docs/group__sparse__func__info.htm#gae6b553df80e21c174d374e82d8505ba5
    """
    values = AFArray.create_null_pointer()
    row_idx = AFArray.create_null_pointer()
    col_idx = AFArray.create_null_pointer()
    storage_id = ctypes.c_int(0)
    call_from_clib(
        sparse_get_info.__name__,
        ctypes.pointer(values),
        ctypes.pointer(row_idx),
        ctypes.pointer(col_idx),
        ctypes.pointer(storage_id),
        arr,
    )
    return (values, row_idx, col_idx, Storage(storage_id.value))


def sparse_get_nnz(arr: AFArray, /) -> int:
    """
    source: https://arrayfire.org/docs/group__sparse__func__nnz.htm#ga0c1ad61d829c02a280c28820eb91f03e
    """
    out = CDimT(0)
    call_from_clib(sparse_get_nnz.__name__, ctypes.pointer(out), arr)
    return out.value


def sparse_get_row_idx(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__row__idx.htm#ga12af820b694c32b1e81fd246ccc87d1e
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sparse_get_row_idx.__name__, ctypes.pointer(out), arr)
    return out


def sparse_get_storage(arr: AFArray, /) -> Storage:
    """
    source: https://arrayfire.org/docs/group__sparse__func__storage.htm#ga31299482afe241ef045f4d107033c999
    """
    out = ctypes.c_int(0)
    call_from_clib(sparse_get_storage.__name__, ctypes.pointer(out), arr)
    return Storage(out.value)


def sparse_get_values(arr: AFArray, /) -> AFArray:
    """
    source: https://arrayfire.org/docs/group__sparse__func__values.htm#ga4d913251cdbedbc48409c88ebddbe2fe
    """
    out = AFArray.create_null_pointer()
    call_from_clib(sparse_get_values.__name__, ctypes.pointer(out), arr)
    return out
