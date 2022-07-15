from gym import envs

for e in envs.registry.all():
    print(e.id)
