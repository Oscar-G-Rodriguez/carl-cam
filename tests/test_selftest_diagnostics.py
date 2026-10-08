from carl_cam.selftest import failure_codes


def test_native_failure_codes_identify_modules_without_exception_text_or_paths():
    try:
        raise ModuleNotFoundError('Sensitive message C:/private/user.txt', name='torch.example')
    except ModuleNotFoundError as error:
        codes = failure_codes(error)
    assert codes['error_type'] == 'ModuleNotFoundError'
    assert codes['missing_module'] == 'torch.example'
    assert codes['failure_frames']
    text = str(codes)
    assert 'Sensitive' not in text and 'private' not in text and 'C:/' not in text
    codes = failure_codes(ModuleNotFoundError('message', name='C:/private/user.txt'))
    assert 'missing_module' not in codes
