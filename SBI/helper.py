
def check_array_length(arr1, arr2, custom_msg=None):
    if len(arr1) != len(arr2):
        if custom_msg is None:
            raise ValueError("Arrays are not of the same length")
        else:
            raise ValueError(custom_msg)