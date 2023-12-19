from angle_emb import AnglE

angle = AnglE.from_pretrained('/home/zhangwc/models/UAE-Large-V1', pooling_strategy='cls').cuda()
vec = angle.encode('hello world', to_numpy=True)
print(vec)
vecs = angle.encode(['hello world1', 'hello world2'], to_numpy=True)
print(vecs)