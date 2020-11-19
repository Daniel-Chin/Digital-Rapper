##### cell 0 #####
## Basics
##### cell 1 #####
from IPython.display import Audio
from numpy import *
import numpy as np
from random import randint
import librosa
import librosa.display
import matplotlib.pyplot as plt
from librosa.util import normalize
sr=22050

def sinw(f, T = .2, phase = 0.):
    global sr
    return sin(2*pi*f*arange(phase, T + phase, 1./sr))

def play(y, sr = sr):
    return Audio(data=y, rate=sr)

def beat2t(beat):
    return beat * 2**14 / sr
##### cell 2 #####
## Our Tools
##### cell 3 #####
def conc(*args):
    chunk = array([])
    for i in args:
        chunk = append(chunk, i)
    return chunk

def pitch2freq(pitch):
    letter=pitch[0].lower()
    height=int(pitch[1])
    note=(height-4)*12 + {'c':0,\
                          'd':2,\
                          'e':4,\
                          'f':5,\
                          'g':7,\
                          'a':9,\
                          'b':11}[letter]
    return int(2**(note/12)* 261.6255653005986)

def sinPitch(pitch, T):
    return sinw(pitch2freq(pitch), T = T)

def mix(plate, *stuff):
    delta = len(stuff[0]) - len(plate)
    if delta < 0:
        new_plate = plate + append(stuff[0], zeros(-delta))
    else:
        new_plate = plate + stuff[0][:len(plate)]
    if len(stuff) == 1:
        return new_plate
    else:
        return mix(new_plate, *stuff[1:])

def am(plate, stuff):
    delta = len(stuff) - len(plate)
    if delta < 0:
        return plate * append(stuff, zeros(-delta))
    else:
        return plate * stuff[:len(plate)]

def syn(func, score, T = .7):
    chunks = []
    for chord in score:
        if chord[0] == 't':
            t = T * int(chord[1])
            chord = chord[2:]
        elif chord[0] == '_':
            t = T / int(chord[1])
            chord = chord[2:]
        else:
            t = T
        chunk = zeros(int(sr * t))
        for pitch in (chord[i:i+2] for i in range(0, len(chord), 2)):
            chunk = mix(chunk, func(pitch, T = t))
        chunks.append(chunk)
    return conc(*chunks)
##### cell 4 #####
### Demo
##### cell 5 #####
play(syn(sinPitch, ('c4','e4','g4','c5', 'c4e4g4c5')))
##### cell 6 #####
## Voice
##### cell 7 #####
def gus(tone, T = .3):
    gmin1 =  [cos((2*pi*tone(n/T)*T) + 3.4 * cos(2*pi*tone(n/T)*T/4)) * (sin(12*pi*n) + 18) for n in arange(0., T, 1./sr)]
    gmin2 =  [cos((2*pi*tone(n/T)*T*4) + 9 * cos(2*pi*tone(n/T)*T/4)) * (sin(12*pi*n) + 2) for n in arange(0., T, 1./sr)]
    fade_in = 1/16
    fade_out = 1/8
    head = exp(linspace(-5., 0., int(fade_in * T * sr)))
    mid = ones(int(T * sr * (1 - fade_in - fade_out)))
    tail = exp(linspace(0., -5., int(fade_out * T * sr)))
    envelope = conc(head, mid, tail)
    return am(add(gmin1, gmin2), envelope)

def boy(tone, T = .3):
    gmin1 =  [cos((2*pi*tone(n/T)*T) + 1.4 * cos(2*pi*tone(n/T)*T/4)) * (sin(12*pi*n) +
                                            13) for n in arange(0., T, 1./sr)]
    gmin2 =  [cos((2*pi*tone(n/T)*T*4) + 13 * cos(2*pi*tone(n/T)*T/4)) * (sin(12*pi*n) +
                                        2) for n in arange(0., T, 1./sr)]
    fade_in = 1/16
    fade_out = 1/8
    head = exp(linspace(-5., 0., int(fade_in * T * sr)))
    mid = ones(int(T * sr * (1 - fade_in - fade_out)))
    tail = exp(linspace(0., -5., int(fade_out * T * sr)))
    envelope = conc(head, mid, tail)
    return am(add(gmin1, gmin2), envelope)

def tone1(x):
    return x*688

def tone2(x):
    sep = .5
    start = 488
    end = 688
    a = (end - start) / sep / 2
    if x < sep:
        return a * x**2 + start * x
    else:
        return (x-sep) * end + a * sep**2 + start * sep

def tone2_high(x):
    sep = .5
    start = 488
    end = 602
    a = (end - start) / sep / 2
    if x < sep:
        return a * x**2 + start * x
    else:
        return (x-sep) * end + a * sep**2 + start * sep

def tone2_(x):
    return x * 602

def tone4(x):
    a = -200
    b = 760
    return a * x**2 + b * x

def tone3_(x):
    start = 265
    end = 396
    return (end - start)/2 * x**2 + start * x

def tone3(x):
    sep = .8
    start = 265
    end = 396
    if x < sep:
        return x * start
    else:
        return (x-sep) * end + sep * start
##### cell 8 #####
### Demo
##### cell 9 #####
play(conc(gus(tone1),gus(tone2),gus(tone3),gus(tone4)))
##### cell 10 #####
# ?????????????????????????????????
play(conc(gus(tone2, .25), 
         gus(tone1, .4), 
         gus(tone4, .2), 
         gus(tone4, .15), 
         gus(tone3), 
         gus(tone1), 
         zeros(int(sr * .3)), 
         boy(tone1, .2), 
         boy(tone2_, .2), 
         boy(tone1, .2), 
         boy(tone2, .3)))
##### cell 11 #####
## Controled Aliasing
##### cell 12 #####
def alia_sin(pitch, T = .7):
    return sinw(440 + sr*randint(10000000000, 200000000000), T)
##### cell 13 #####
### Demo
##### cell 14 #####
noisy_fart = syn(alia_sin, ('g4c5e5','g4c5','e4g4','e4c5', 'g4b4d5', 't2b4d5g5', 'g4b4d5','a4c5e5','a4c5d5','b3d4g4','e4g4c5','t4c3c4g4c5'), .6)
play(noisy_fart)
##### cell 15 #####
play(syn(alia_sin, ('a4c5e5','a4c5f5','g4c5e5', 'g4b4d5'), T = 2))
##### cell 16 #####
## Bit-wise
##### cell 17 #####
def gusBitW(n):
    return n & n>>8

def danBitW(n, b, c):
    return n & (n>>b | n>>c)

def duck(frames):
    return cos((arange(frames)/frames-.0)*pi*16)*.4 + .6

chunk = []
for c in range(1,14):
    chunk += [min(danBitW(n, 11, c), 1) for n in arange(32768)]
for c in range(1,12):
    chunk += [min(danBitW(n, 13, c), 1) for n in arange(32768)]
dan_bit_w = am(chunk, tile(duck(32768), 13 + 11))
# Daniel: I exported dan_bit_w to Audacity and changed the sample rate, 
# so it is the same as the main sr. 
# Here I load it back. 
dan_bit_w, _ = librosa.load('dan.wav')

# Bit wise tracks that Matthew discovered
math_music, _ = librosa.load('Math_bw_music.wav')
math_music = math_music[:2**17]

# Formulas modified from viznut. We used http://wurstcaptures.untergrund.net/music/ to generate sounds. 
# Formula: (((t*t%256)|(t>>4))>>1)|(t>>3)|(t<<2|t)
nut, _ = librosa.load('nut.wav')
nut = mix(zeros(2**15), nut)

# (t*(t>>8*(t>>2|t>>8)&(20|(t>>19)*5>>t|t>>3)))
hexa, _ = librosa.load('Hexa.wav')
hexa = mix(zeros(2**19), hexa)

# (t*(t>>8*(t>>11|t>>8)&(20|(t>>19)*5>>t|t>>3)))
step, _ = librosa.load('Step.wav')
step = mix(zeros(2**19), step)

# (t*(t>>8*(t>>15|t>>8)&(20|(t>>19)|t>>2)))
ut, _ = librosa.load('UT.wav')
ut = mix(zeros(2**17), ut)
##### cell 18 #####
### Demo
##### cell 19 #####
play(math_music)
##### cell 20 #####
## Ring
##### cell 21 #####
def ring(pitch, T):
    # A ring sound
    Fc = pitch2freq(pitch)
    Fm = Fc * 1.25
    gmin =  [sin(2*pi*Fc*n) * sin(2*pi*Fm*n) * exp(-2*n/T) for n in arange(0.,T,1./sr)]
    return gmin

fart = syn(ring, ('g4e5','c4c5','g3g4','c4c5', 'b4d5', 't2g4g5', 'd4d5','c5e5','d4d5','g3g4','t5c4g4c5'), .3)
play(fart)
##### cell 22 #####
## All is ready. Start to compose! 
##### cell 23 #####
greet = conc(gus(tone2, .25), 
         gus(tone1, .4), 
         gus(tone4, .2), 
         gus(tone4, .15), 
         gus(tone3), 
         gus(tone1), 
         zeros(int(sr * .3)), 
         gus(tone1, .2), 
         gus(tone2_, .2), 
         gus(tone1, .2), 
         gus(tone2, .3))
##### cell 24 #####
# ??????????????????????????????????????????????????????????????????????????????
def maolaoshu(t):
    mao = gus(tone1, t)
    laoshu = append(gus(tone2, t/2), gus(tone3_, t/2))
    da = gus(tone4, t/2)
    lobo = append(gus(tone2, t/4), gus(tone1, t/4))
    pilipala = conc(*[gus(tone1, t/4)]*4)
    ddp = conc(
        gus(tone4, t/3), 
        gus(tone1, t/3), 
        gus(tone4, t/3))
    qnd = conc(
        gus(tone4, t/4), 
        gus(tone3_, t/2), 
        gus(tone2_, t/4))
    return conc(mao, laoshu, laoshu, mao, da, lobo, lobo, da, laoshu, 
                mao, pilipala, ddp, qnd, mao, zeros(int(t * sr * 4)))
mls = None
mls = append(maolaoshu(beat2t(1)), maolaoshu(beat2t(.5)))
mls = mix(normalize(mls), step)
play(mls)
##### cell 25 #####
# Rap + dan_bit
t = beat2t(.5 / .8)
rap = conc(gus(tone1, t), 
    gus(tone2, t), 
    gus(tone3, t), 
    gus(tone1, t/2), 
    gus(tone2, t/2), 
    gus(tone1, t/2), 
    gus(tone1, t/2), 
    gus(tone2, t), 
    gus(tone3, t), 
    gus(tone1, t/2), 
    gus(tone2, t/2), 
    gus(tone1, t/2), 
    gus(tone2, t/2), 
    gus(tone3, t), 
    gus(tone2, t/2), 
    gus(tone2, t/2), 
    gus(tone1, t), 
    gus(tone2, t), 
    zeros(int(sr * t)), 
    gus(tone2, t), 
    zeros(int(sr * t)), 
    gus(tone4, t), 
    gus(tone2, t), 
    gus(tone4, t), 
    gus(tone2, t), 
    gus(tone3, t/2), 
    gus(tone2, t), 
    gus(tone2, t/2), 
    gus(tone2, t), 
    gus(tone3, t), 
    gus(tone4, t/2), 
    gus(tone4, t/2), 
    gus(tone1, t), 
    gus(tone4, t/2), 
    gus(tone4, t/2), 
    gus(tone4, t), 
    gus(tone2, t/2), 
    gus(tone2, t/2), 
    gus(tone2, t/4), 
    gus(tone2, t/4), 
    gus(tone2, t/4), 
    gus(tone2, t/4), 
    gus(tone2, t), 
    zeros(int(sr * t)))
dan_rap = mix(dan_bit_w[:len(dan_bit_w)//2], normalize(tile(rap, 2))/2)
play(dan_rap)
##### cell 26 #####
# ???????????????
score = (
    'd4', 'c4', 
    't2e4', 'e4', 't2e4', 't2e4', 't7e4', 'd4', 'c4', 
    't2d4', 'd4', 't2d4', 't2e4', 't7e4', 'a3', 'b3', 
    't2c4', 'c4', 't3c4', 't2d4', 't2e4', 'c4', 't3c4', 'b3', 'a3'
    'e3', 'b3', 'e4', 'e4', 'e4', 't2e4', 'g4', 't9e4', 
    'a3', 'f4', 'f4', 'f4', 't2f4', 'a3', 't2f4', 't2f4', 'f4', 't4f4', 
    'b3', 'g4', 'g4', 'g4', 't2g4', 'b3', 't5g4', 't4d4'
)
hbx = syn(ring, score, T = beat2t(.25))
play(hbx)
##### cell 27 #####
t = beat2t(.25)
chunk = []
chunk.append(conc(
    boy(tone3_, 2*t),#?????????
    boy(tone1, t),
    boy(tone4, 3*t),
    boy(tone3, t),#???????????????
    boy(tone1, 2*t),
    boy(tone3, t),
    boy(tone4, 2*t),
    boy(tone4, 4*t),
    boy(tone4, t),#????????????
    boy(tone3, t),
    boy(tone4, t),
    boy(tone4, t*3),
    boy(tone2_, t),#?????????
    boy(tone4, t*2),
    boy(tone2, t*6)
))
chunk.append(conc(
    boy(tone4, t),#???????????????
    boy(tone2_high, 2*t),
    boy(tone4, t),
    boy(tone3, t),
    boy(tone4, 2*t), 
    boy(tone3, t),#??????????????????
    boy(tone4, t),
    boy(tone3, t),
    boy(tone1, t),
    boy(tone4, t),
    boy(tone3, t*3),
    boy(tone1, t),#?????????????????????
    boy(tone1, t),
    boy(tone3, t*3),
    boy(tone2_, t),
    boy(tone4, t*2),
    boy(tone1, t),
    boy(tone1, t*7)
))
chunk.append(conc(
    boy(tone3, t),
    boy(tone1, t),
    boy(tone4, t),
    boy(tone3, t),
    boy(tone4, t),
    boy(tone3, t),
    boy(tone4, t),
    boy(tone1, t*2),
    boy(tone3, t),
    boy(tone4, t),
    boy(tone1, t),
    boy(tone2, t),
    boy(tone3, t),
    boy(tone2, t*4)
))
chunk.append(conc(
    boy(tone3, t),
    boy(tone4, t),
    boy(tone4, t),
    boy(tone2, t),
    boy(tone4, t),
    boy(tone2, t),
    boy(tone4, 2*t),
    boy(tone4, t),
    boy(tone2, t),
    boy(tone3, t),
    boy(tone4, t),
    boy(tone4, 4*t)
))
chunk.append(conc(
    boy(tone3, t*2),
    boy(tone4, t),
    boy(tone2, t),
    boy(tone4, t*2),
    boy(tone3, t),
    boy(tone4, t),
    boy(tone2, t),
    boy(tone3, t*2),
    boy(tone2, t),
    boy(tone1, t*4),
    boy(tone1, t*3),
    boy(tone2, t),
    boy(tone3, t*10)
))
chunk.append(tile(conc(
    boy(tone2, t),#?????? ??????
    boy(tone1, t),
    boy(tone2, t*2),
    boy(tone1, t*4),
    boy(tone4, t),#?????????????????????
    boy(tone4, t),
    boy(tone1, t*2),
    boy(tone2, t),
    boy(tone4, t),
    boy(tone4, t*2),
    boy(tone4, t*2),
    boy(tone2, t),#??????
    boy(tone1, t*3),
    zeros(int(sr * t*4)),
    gus(tone2, t),#??????
    gus(tone1, t*3),
    zeros(int(sr * t*2))
), 2))
pangmailang = conc(*chunk)
play(pangmailang)
##### cell 28 #####
# Snowdin by Toby Fox
score = (
    'g5', 'g4', 'c5', 'g4', 't3d5', '_2c5', '_2d5',
    'e5', 'c5', 'g4', 't2e4', 'f4', 't2g4', 
    'g5', 'g4', 'c5', 'g4', 't3d5', '_2c5', '_2d5',
    'e5', 'f5', 'e5', 't2d5', 'e5', 't2c5'
)
snowdin = syn(ring, score, .3)
play(snowdin)
##### cell 29 #####
our_song = []
our_song += normalize(fart).tolist()
greet_ = mix(fart, normalize(greet))
our_song += normalize(greet_).tolist()
our_song += (normalize(step[:int(sr * beat2t(4))])/2).tolist()
our_song += normalize(mls).tolist()
our_song += normalize(mix(tile(math_music, 2), step)).tolist()
our_song += normalize(mix(math_music, ut)).tolist()
our_song += normalize(dan_rap).tolist()
our_song += normalize(hexa[:-int(beat2t(4)*sr)]).tolist()
our_song += normalize(mix(hexa[-int(beat2t(4)*sr):], tile(nut,4))).tolist()
our_song += normalize(hbx).tolist()
pang_ = mix(normalize(pangmailang), append(tile(nut, 16), step))
our_song += normalize(pang_).tolist()
our_song += normalize(fart).tolist()
our_song += normalize(mix(tile(fart, 2), snowdin)).tolist()
our_song += normalize(snowdin).tolist()
our_song += normalize(noisy_fart).tolist()
librosa.output.write_wav('our.wav', array(our_song), sr)
##### cell 30 #####
## Done!
##### cell 31 #####
#### Somehow, my jupyter notebook refuses to load the music. Probably too big for RAM. Please check the wav file instead. 
