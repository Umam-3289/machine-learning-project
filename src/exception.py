import sys

def error_meassage_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occurred in python Script name [{0}] in line no. [{1}] error meassage[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error))
    
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().error_message(error_message)
        self.error_message=error_meassage_detail(error_message,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message