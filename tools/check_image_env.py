import importlib, json, sys
out = {}
modules = [('torch','torch'), ('open_clip','open_clip'), ('PIL','PIL')]
for key, modname in modules:
    try:
        mod = importlib.import_module(modname)
        if key == 'torch':
            out['torch'] = True
            out['torch_version'] = getattr(mod, '__version__', 'unknown')
            out['cuda_available'] = getattr(mod, 'cuda', None) and mod.cuda.is_available()
        elif key == 'open_clip':
            out['open_clip'] = True
            out['open_clip_version'] = getattr(mod, '__version__', 'unknown')
        else:
            out['Pillow'] = True
            out['Pillow_version'] = getattr(mod, '__version__', 'unknown')
    except Exception as e:
        out[key] = False
print(json.dumps(out))
sys.exit(0)
