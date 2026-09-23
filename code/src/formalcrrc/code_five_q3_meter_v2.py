"""Use supported named worker RPCs; never enable pickle fallback serialization."""
from formalcrrc.code_five_q3_meter import install_meter,read_meter,metered_class

class FiveMeterWorkerExtension:
    def five_install_meter(self):return install_meter(self)
    def five_read_meter(self):return read_meter(self)

def named_rpc_metered_class(base,root):
    class NamedRPCLLM(base):
        def __init__(self,*args,**kwargs):
            kwargs['worker_extension_cls']='formalcrrc.code_five_q3_meter_v2.FiveMeterWorkerExtension'
            super().__init__(*args,**kwargs)
        def collective_rpc(self,method,*args,**kwargs):
            if method is install_meter:method='five_install_meter'
            elif method is read_meter:method='five_read_meter'
            return super().collective_rpc(method,*args,**kwargs)
    return metered_class(NamedRPCLLM,root)
