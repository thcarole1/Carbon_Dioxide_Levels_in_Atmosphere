def say_hello():
    print('Hello World !')

if __name__ == '__main__':
    try:
       say_hello()

    except:
        import sys
        import traceback
        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
