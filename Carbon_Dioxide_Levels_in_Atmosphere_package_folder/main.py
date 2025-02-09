def say_hello():
    print('Hello World !')

def predict_co2_level():
    return 'nothing to show'

if __name__ == '__main__':
    try:
       predict_co2_level()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
