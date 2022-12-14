def get_preprocessor_class(name):
    name = 'preprocessing.' + name
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
