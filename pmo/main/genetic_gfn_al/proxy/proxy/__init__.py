from proxy.proxy.regression import DropoutRegressor, EnsembleRegressor

def get_proxy_model(tokenizer,num_token,max_len,args,category='dropout'):

    if category=='dropout':
        proxy = DropoutRegressor(tokenizer,num_token,max_len,args)
    else:
        proxy = EnsembleRegressor(tokenizer,num_token,max_len)
    return proxy