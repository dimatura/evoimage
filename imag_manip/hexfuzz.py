
import random

im = open('hexedit.raw', 'rb')
bytes = im.read()
im.close()
out = open('hexout.raw', 'wb')

outbytes = []
for i,b in enumerate(bytes):
    if i < 100:
        outbytes.append(b)
        continue
    if random.uniform(0., 1.) > .9997:
        s = random.choice(bytes)
        outbytes.append(s)
        s = random.choice(bytes)
        outbytes.append(s)
        s = random.choice(bytes)
        outbytes.append(s)
        s = random.choice(bytes)
        outbytes.append(s)
    if random.uniform(0., 1.) > .999:
        continue
    if random.uniform(0., 1.) < .999:
        outbytes.append(b)


outbytes = ''.join(outbytes)

out.write(outbytes)
out.close()
