#-----------------------------------------------------------------------------------------------
# Initialize class default arguments automatically (instead of doing it one by one in __init__)
#-----------------------------------------------------------------------------------------------
def initFromArgs(beingInitted, bJustArgs=False):
    import sys
    codeObject = beingInitted.__class__.__init__.im_func.func_code
    for k,v in sys._getframe(1).f_locals.items():
        if k!='self' and ((not bJustArgs) or k in codeObject.co_varnames[1:codeObject.co_argcount]):
            setattr(beingInitted,k,v)
