import sys


def error_message_detail(error, error_details: sys):
    _, _, error_tb = error_details.exc_info()
    file_name = error_tb.tb_frame.f_code.co_filename
    line_number = error_tb.tb_lineno
    error_msg = str(error)
    return f"Error in {file_name} , line :{line_number} : {error_msg}"


class CustomException(Exception):
    def __init__(self, error, error_details: sys):
        super().__init__(error)
        self.error_message = error_message_detail(error, error_details)

    def __str__(self):
        return self.error_message

