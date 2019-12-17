# Module: ecen4242f19.py
# ECEN 4242 Fall 2019, Functions for communication signals

import numpy as np
import scipy.signal as ss


def asc2bin(txt, bits):
    """
    ASCII character sequence to binary string conversion.
    >>>>> dn = asc2bin(txt, bits) <<<<<
    where  dn       binary output string
           txt      text input (ASCII)
           bits<0   MSB first conversion
           bits>=0  LSB first conversion
           |bits|   number of bits per character
    """
    txt_10 = list(ord(chr) for chr in txt)
    if (bits < 0):
        pow2 = list(2**(i+1) for i in range(bits,0))
    else:
        pow2 = list((2**-i) for i in range(bits))
    B = np.array(np.outer(txt_10, pow2), int)
    B = np.mod(B, 2)
    dn = np.reshape(B, -1)
    return dn
    
    
def bin2asc(dn, bits):
    """
    Binary string to ASCII character string conversion.
    >>>>> txt = bin2asc(dn, bits) <<<<<
    where  txt      output string (ASCII)
           dn       binary string
           bits<0   MSB first conversion
           bits>=0  LSB first conversion
           |bits|   number of bits per character
    """
    Lb = int(np.floor(len(dn)/abs(bits)))  # length in multiples of 'bits'
    dn = dn[:Lb*abs(bits)]
    B = np.reshape(dn, (-1, abs(bits)))
    if (bits < 0):
        pow2 = list(2**(i-1) for i in range(abs(bits),0,-1))
    else:
        pow2 = list(2**i for i in range(bits))
    txt_10 = np.inner(B, pow2)
    return ''.join(chr(i) for i in txt_10) 
    
    
def b2M(dn, m=2):
    """
    Bits to M-ary (M=2^m) symbols conversion
    >>>>> sn = b2M(dn, m) <<<<<
    """
    M = 2**m    # number of symbol values
    dL = int(m*np.ceil(len(dn)/float(m))-len(dn))
    dn = np.append(dn, np.zeros(dL))
    B = np.reshape(dn, (-1, m))
    pow2 = list(2**(i-1) for i in range(m,0,-1))
    sn = np.inner(B, pow2)
    return sn
    
    
def M2b(sn, m=2):
    """
    M-ary (M=2^m) symbols to binary conversion
    >>>>> dn = M2b(sn, m) <<<<<
    """
    pow2 = list(2**(i+1) for i in range(-m,0))
    B = np.array(np.outer(sn, pow2), int)
    B = np.mod(B, 2)
    sn = np.reshape(B, -1)
    return sn
    

def pam_pt(FB, Fs, ptype, pparms=[]):
    """
    Generate PAM pulse p(t)
    >>>>> ttp, pt = pam_pt(FB, Fs, ptype, pparms) <<<<<
    where  ttp:   time axis for p(t)
           pt:    PAM pulse p(t)
           FB:    Baud rate  (Fs/FB=sps)
           Fs:    sampling rate of p(t)
           ptype: pulse type from list
                  ('man', 'msin', rcf', 'rect', 'rrcf', 'sinc', 'tri')
           pparms not used for 'man','msin','rect','tri'
           pparms = [k, alfa]  for 'rcf', 'rrcf'
           pparms = [k, beta]  for 'sinc'
           k:     "tail" truncation parameter for 'rcf','rrcf','sinc'
                  (truncates p(t) to -k*TB <= t < k*TB)
           beta:  Kaiser window parameter for 'sinc'
           alfa: Rolloff parameter for 'rcf','rrcf', 0<=alfa<=1
    """
    ptyp = ptype.lower()
    if (ptyp=='rect' or ptyp=='man' or ptyp=='msin'):
        kR = 0.5; kL = -kR
    elif ptyp=='tri':
        kR = 1.0; kL = -kR
    elif (ptyp=='rcf' or ptyp=='rrcf' or ptyp=='sinc'):
        kR = pparms[0]; kL = -kR
    else:
        kR = 0.5; kL = -kR
    tpL, tpR = kL/float(FB), kR/float(FB)
    ixpL, ixpR = int(np.ceil(tpL*Fs)), int(np.ceil(tpR*Fs))
    ttp = np.arange(ixpL, ixpR)/float(Fs)  # time axis for p(t)
    pt = np.zeros(ttp.size)
    if ptyp=='man':
        pt = -np.ones(ttp.size)
        ixp = np.where(ttp>=0)
        pt[ixp] = 1
    elif ptyp=='msin':
        pt = np.sin(2*np.pi*FB*ttp)
    elif ptyp=='rcf':
        pt = np.sinc(FB*ttp)
        if pparms[1] != 0:
            p2t = np.pi/4.0*np.ones(ttp.size)
            ix = np.where(np.power(2*pparms[1]*FB*ttp, 2.0) != 1)[0]
            p2t[ix] = np.cos(np.pi*pparms[1]*FB*ttp[ix])
            p2t[ix] = p2t[ix]/(1-np.power(2*pparms[1]*FB*ttp[ix],2.0))
            pt = pt*p2t
    elif ptyp=='rect':    
        ixp = np.where(np.logical_and(ttp>=tpL,ttp<tpR))[0]
        pt[ixp] = 1    # rectangular pulse p(t)
    elif (ptype=='rrcf'):      # Root raised cosine in freq
        alfa = pparms[1]       # Rolloff parameter
        falf = 4*alfa*FB
        pt = (1-alfa+4*alfa/np.pi)*np.ones(len(ttp))
        ix = np.where(np.logical_and(ttp!=0,np.power(falf*ttp,2.0)!=1.0))[0]
        pt[ix] = np.sin((1-alfa)*np.pi*FB*ttp[ix])
        pt[ix] = pt[ix]+falf*ttp[ix]*np.cos((1+alfa)*np.pi*FB*ttp[ix])
        pt[ix] = 1.0/(FB*np.pi)*pt[ix]/((1-np.power(falf*ttp[ix],2.0))*ttp[ix])
        ix = np.where(np.power(falf*ttp,2.0)==1.0)[0]
        pt[ix] = (1+2/np.pi)*np.sin(np.pi/(4*alfa))+(1-2/np.pi)*np.cos(np.pi/(4*alfa))
        pt[ix] = alfa/np.sqrt(2.0)*pt[ix]
    elif ptyp=='sinc':
        pt = np.sinc(FB*ttp)
        if len(pparms) > 1:        # Apply Kaiser window 
            pt = pt*np.kaiser(len(pt),pparms[1])
    elif ptyp=='tri':
        pt = 1 + FB*ttp
        ixp = np.where(ttp>=0)[0]
        pt[ixp] = 1 - FB*ttp[ixp]
    else:
        ix0 = np.argmin(np.abs(ttp))
        pt[ix0] = 1
    return ttp, pt
    
    
def pam15(an, FB, Fs, ptype, pparms=[]):
    """
    Pulse amplitude modulation: a_n -> s(t), -TB/2<=t<(N-1/2)*TB,
    V1.5 for 'man', 'msin', 'rcf', 'rect', 'rectx', 'rrcf', 'sinc', and
    'tri' pulse types.
    >>>>> tt, st = pam15(an, FB, Fs, ptype, pparms) <<<<<
    where  tt:    time axis for PAM signal s(t) (starting at -TB/2)
           st:    PAM signal s(t)
           an:    N-symbol DT input sequence a_n
           FB:    baud rate of a_n, TB=1/FB
           Fs:    sampling rate of s(t)
           ptype: pulse type from list
                  ('man','rcf','rect','rrcf','sinc','tri')
           pparms not used for 'man','rect','tri'
           pparms = [k, alpha] for 'rcf','rrcf'
           pparms = [k, beta]  for 'sinc'
           k:     "tail" truncation parameter for 'rcf','rrcf','sinc'
                  (truncates p(t) to -k*TB <= t < k*TB)
           alpha: Rolloff parameter for 'rcf','rrcf', 0<=alpha<=1       
           beta:  Kaiser window parameter for 'sinc'
    """
    N = len(an)
    ixL = round(-0.5*Fs/float(FB))    # Left index for time axis
    tlen = N/float(FB)   # duration of PAM signal in sec
    tt = np.arange(round(Fs*tlen))/float(Fs)
    tt = tt + ixL/float(Fs)   # shift time axis left by TB/2
    if ptype.lower() == 'rectx':
        ixa = np.array(np.round(Fs/float(FB)*np.arange(N)),np.int64)
        st = np.zeros(tt.size)
        st[ixa] = Fs*np.diff(np.hstack((0, an)));   # place transitions in s(t)
        st = np.cumsum(st)/float(Fs)
    else:
        ixa = np.array(np.round(Fs/float(FB)*(0.5+np.arange(N))),np.int64)
        ast = np.zeros(tt.size)
        ast[ixa] = Fs*an   # as(t) is CT version of an
        ttp, pt = pam_pt(FB, Fs, ptype, pparms)
        # Convolution  as(t)*p(t)
        st = np.convolve(ast, pt)/float(Fs)  # s(t) = a_s(t)*p(t)
        ixttp0 = np.argmin(np.abs(ttp))  # index for t=0 on ttp 
        st = st[ixttp0:]  # trim after convolution
        st = st[:tt.size]  # PAM signal s(t)
    return tt, st
    
    
def pamrcvr15(tt, rt, FBparms, ptype, pparms=[]):
    """
    Pulse amplitude modulation receiver with matched filter:
    r(t) -> b(t) -> bn. 
    V1.5 for 'man', 'msin', 'rcf', 'rect', 'rrcf', 'sinc', and 'tri'
    pulse types.
    >>>>> bn, bt, ixn = pamrcvr15(tt, rt, FBparms, ptype, pparms) <<<<<
    where  tt:    time axis for r(t)
           rt:    received (noisy) PAM signal r(t)
           FBparms: = [FB, dly]
           FB:    Baud rate of PAM signal, TB=1/FB
           dly:   sampling delay for b(t) -> b_n as a fraction of TB
                  sampling times are t=n*TB+t0 where t0 = dly*TB
           ptype: pulse type from list
                  ('man','msin','rcf','rect','rrcf','sinc','tri')
           pparms not used for 'man','msin','rect','tri'
           pparms = [k, alpha]  for 'rcf','rrcf'
           pparms = [k, beta]  for 'sinc'
           k:     "tail" truncation parameter for 'rcf','rrcf','sinc'
                  (truncates p(t) to -k*TB <= t < k*TB)
           alpha: rolloff parameter for ('rcf','rrcf'), 0<=alpha<=1
           beta:  Kaiser window parameter for 'sinc'
           bn:    received DT sequence after sampling at t=n*TB+t0
           bt:    received PAM signal b(t) at output of matched filter
           ixn:   indexes where b(t) is sampled to obtain b_n
    """
    if type(FBparms)==int:
        FB, t0 = FBparms, 0
    else:    
        FB, t0 = FBparms[0], 0
        if len(FBparms) > 1:
            t0 = FBparms[1]
    Fs = (len(tt)-1)/(tt[-1]-tt[0])
    # ***** Set up matched filter response h_R(t) *****
    ttp, pt = pam_pt(FB, Fs, ptype, pparms)
    hRt = pt[::-1]             # h_R(t) = p(-t)
    hRt = Fs/np.sum(np.power(pt,2.0))*hRt  # h_R(t) normalized
    # Convolution  r(t)*h_R(t)
    bt = np.convolve(rt, hRt)/float(Fs)  # b(t) = r(t)*h_R(t)
    ixttp0 = np.argmin(np.abs(ttp))  # index for t=0 on ttp 
    bt = bt[ixttp0:]  # trim after convolution
    bt = bt[:tt.size]  # PAM signal b(t) after matched filter
    N = np.ceil(FB*(tt[-1]-tt[0]))    # Number of symbols
    ixn = np.array(np.around((np.arange(N)+0.5+t0)*Fs/FB),int)
                               # Sampling indexes
    ix = np.where(np.logical_and(ixn>=0,ixn<len(tt)))[0]
    ixn = ixn[ix]              # Trim to existing indexes
    bn = bt[ixn]               # DT sequence sampled at t=n*TB+t0
    return bn, bt, ixn
    
    
def FTapprox(tt, xt, ff_lim=[]):
    """
    Fourier transform X(f) approximation to waveform x(t), using DFT/FFT
    >>>>> ff, absXf, argXf, Df = FTapprox(tt, xt, ff_lim) <<<<<
    where  ff:   frequency axis in Hz
           absXf:  magnitude of X(f)
           argXf:  phase of X(f) in degrees
           Df:   frequency resolution in Hz
           ff_lim = [f1, f2, llim]
           f1:   start frequency
           f2:   end frequency
           llim=0: linear magnitude, unmodified phase
           llim=thr>0: linear magnitude, phase=0 if |X(f)|<thr
           llim=thr<0: magnitude in dB (relative to max(|X(f)|), lower limit
                       thr (in dB), phase=0 if |X(f)| in dB below thr
    """
    Fs = (len(tt)-1)/float(tt[-1]-tt[0])  # sampling rate
    ixp = np.where(tt>=0)[0]
    ixn = np.where(tt<0)[0]
    Xf = np.fft.fft(np.hstack((xt[ixp], xt[ixn])))/float(Fs)
    N = Xf.size
    if len(ff_lim)==0:
        ff_lim = [-Fs/2.0, Fs/2.0, 0]  # default values
    if len(ff_lim)<2:
        ff_lim = [ff_lim[0], Fs/2.0, 0]
    if len(ff_lim)<3:
        ff_lim = [ff_lim[0], ff_lim[1], 0]
    Df = Fs/float(N)   # frequency resolution
    ff = Df*np.arange(N)   # frequency axis
    if ff_lim[0]<0:
        ff = ff-Fs/2.0
        Xf = np.fft.fftshift(Xf)
    Xfmax = np.max(np.abs(Xf))    # max(|X(f)|)
    ixfd = np.where(np.logical_and(ff>=ff_lim[0], ff<ff_lim[1]))[0]
    ff = ff[ixfd]  # trim do display limits 
    Xf = Xf[ixfd]
    absXf = np.abs(Xf)   # magnitude of X(f)
    argXf = 180/np.pi*np.angle(Xf)   # phase of X(f)
    if ff_lim[2]<0:
        absXf = absXf/Xfmax   # normalized magnitude
        llim = 10**(ff_lim[2]/20.0)
        ixm = np.where(absXf<llim)
        absXf[ixm] = llim   # set lower limit
        argXf[ixm] = 0   # mask phase
        absXf = 20*np.log10(absXf)  # normalized magnitude in dB
    elif ff_lim[2]>0:
        ixm = np.where(absXf<ff_lim[2])
        argXf[ixm] = 0   # mask phase
    return ff, absXf, argXf, Df     
    
    
def eyediagram(tt, rt, FB, dispparms=[]):
    """
    Generate waveform array for eye diagram of digital PAM signal r(t)
    >>>>> ttA, A = eyediagram(tt, rt, FB, dispparms) <<<<<
    where  tt:  time axis for rt
           rt:  received PAM signal r(t)=sum_n a_n*q(t-nTB)
           FB:  Baud rate of DT sequence a_n, TB = 1/FB
           dispparms = [NTd, delay, width, step]
           NTd:    Number of traces to display
           delay:  trigger delay (in TB units, e.g., 0.5)
           width:  display width (in TB units, e.g., 3)
           step:   step size from trace to trace (in TB units)
           ttA: time axis (in TB) for eye diagram display
           A:   array of eye diagram traces
    """
    # Parameters
    if type(dispparms)==int:
        dispparms = [dispparms]
    if len(dispparms)==0:
        dispparms = [50]   # default # of traces
    if len(dispparms)==1:
        dispparms = np.hstack((dispparms, 0))  # default delay
    if len(dispparms)==2:
        dispparms = np.hstack((dispparms, 3))  # default width
    if len(dispparms)==3:
        dispparms = np.hstack((dispparms, 1))  # default step
    # Setup
    Fs = (len(tt)-1)/(tt[-1]-tt[0])
    NTd = int(dispparms[0])      # Number of traces
    t0 = dispparms[1]/float(FB)  # Delay in sec
    if t0<tt[0]:
        t0 = tt[0]
    tw = dispparms[2]/float(FB)  # Display width in sec
    tstep = dispparms[3]/float(FB)  # Step size in sec
    tend = t0 + NTd*tstep + tw   # End time
    if tend>tt[-1]:
        NTd = int(np.floor((tt[-1]-t0-tw)/tstep))
    ixw = int(round(tw*Fs))        # samples per width
    A = np.zeros((NTd, ixw))    # Array for traces
    ix0 = np.argmin(np.abs(tt)) # index of t=0
    ixd0 = ix0 + int(round(t0*Fs))
    for i in range(NTd):
        ixi = ixd0 + int(round(i*tstep*Fs))
        A[i,:] = rt[ixi:ixi+ixw]
    ttA = FB*np.arange(ixw)/float(Fs)    
    return ttA, A
    
    
def iir2filt(an, xn, Sn=[]):
    """
    Modulo 2 IIR filter of order L = len(an)-1
    >>>>> yn = iir2filt(an, xn, Sn) <<<<<
    where  yn: (binary) output sequence
           an: filter coefficients [a[0]=1, a[1], ..., a[L]]
           xn: (binary) input sequence
           Sn: initial state [S[-L], S[-L+1], ..., S[-1]]
    """
    an = np.array(an, int)  # make array of integers
    L = an.size-1   # filter order
    if len(Sn)<L:
        Sn = np.hstack((Sn, np.zeros(L-len(Sn))))  # pad with zeros if needed
    Sn = np.array(Sn, int)  # make array of integers
    xn = np.array(xn, int)  # make array of integers
    N = xn.size  # length of input sequence
    yn = np.hstack((Sn, np.zeros(N)))  # initialize output sequence
    yn = np.array(yn, int)  # make array of integers
    for n in range(N):
        yn[n+L] = (xn[n] + np.sum(yn[n:n+L]*an[-1:0:-1]))%2
    return yn[L:]   # remove initial state
    
    
def lfsr(M, gn, S0=[]):
    """
    Generate M symbols of binary LFSR sequence with generator
    polynomial gn = [g[0]=1, g[1], ..., g[L]=1]. Default gn is
    used if L is specified instead of gn.
    >>>>> yn = lfsr(M, gn, S0) <<<<<
    where  yn: (binary) output sequence
            M: number of output symbols
           gn: generator polynomial  (default gn if gn=L)
           S0: initial state [S[-L], S[-L+1], ..., S[-1]]
    """
    if type(gn)==int:
        L = gn
        if L==3:
            gn = [1,0,1,1]
        elif L==4:
            gn = [1,0,0,1,1]
        elif L==5:
            gn = [1,0,0,1,0,1]
        elif L==6:
            gn = [1,0,0,0,0,1,1]
        elif L==7:
            gn = [1,0,0,0,0,0,1,1]
        elif L==8:
            gn = [1,1,0,0,0,0,1,1,1]
        elif L==9:
            gn = [1,0,0,0,0,1,0,0,0,1]
        elif L==10:
            gn = [1,0,0,0,0,0,0,1,0,0,1]
        elif L==11:
            gn = [1,0,0,0,0,0,0,0,0,1,0,1]
        elif L==17:
            gn = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1]
        else:
            print('No default gn for L={}'.format(L))
    else:
        L = len(gn)-1
    gn = np.array(gn, int)  # make array of integers
    if len(S0)==0:
        yn = iir2filt(gn, np.hstack((1, np.zeros(M-1))))
    else:
        yn = iir2filt(gn, np.zeros(M), S0)
    return yn
    
    
def BWfilt(xt, parms):
    """
    Brickwall filter with windowing
    >>>>> yt = BWfilt(xt, parms) <<<<<
    where  yt: delay compensated filter output
           xt: filter input, sample rate Fs
           parms=[fL, Fs, k, beta]
           fL: cutoff frequency in Hz
           Fs: sampling rate in Hz
           k:  taillength of h(t)
               -k/2fL <= t < k/2fL
           beta: Kaiser window parameter
    """
    Fs = parms[1]
    fL = parms[0]
    k = 5
    if len(parms)>2:
        k = parms[2]
    ixk = round(Fs*k/(2.0*fL))
    tth = np.arange(-ixk,ixk)/float(Fs)
    ht = 2*fL*np.sinc(2*fL*tth)
    if len(parms)>3:
        ht = ht*np.kaiser(ht.size, parms[3])
    yt = np.convolve(np.hstack((xt, np.zeros(ixk))), ht)/float(Fs)
    yt = yt[ixk:]    # trim to original length
    return yt[:len(xt)]
    
    
def butterLPF(xt, Fs, fparms, dly=0):
    """
    Butterworth lowpass filter with delay compensation, digital design for
    pseudo-CT (continuous time) signals.
    >>>>> yt, ord = butterLPF(xt, Fs, fparms, dly) <<<<<
    where  yt:  (filtered) output signal, sample rate Fs
           ord: actual filter order
           xt:  input signal, sample rate Fs
           Fs:  sample rate of yt, xt in Hz
           fparams = [N, fL]
           N:   filter order
           fL:  cutoff frequency (-3dB) in Hz
           fparams = [fp, fs, gp, gs]
           fp:  passband frequency in Hz
           fs:  stopband frequency in Hz
           gp:  max passband loss in dB
           gs:  min stopband attenuation in dB
           dly: filter delay compensation in sec
    """
    if len(fparms)==4:
        N, wn = ss.buttord(2*fparms[0]/float(Fs),2*fparms[1]/float(Fs),fparms[2],fparms[3])
    else:
        N, wn = fparms[0], 2*fparms[1]/float(Fs)
    sos = ss.butter(N, wn, output='sos')
    fdly = round(dly*Fs)
    yt = ss.sosfilt(sos, np.hstack((xt, np.zeros(fdly))))
    yt = yt[fdly:]    # filter delay compensation
    return yt, N
    
    
def cheby1LPF(xt, Fs, fparms, dly=0):
    """
    Chebyshev type 1 lowpass filter with delay compensation, digital design for
    pseudo-CT (continuous time) signals.
    >>>>> yt, ord = cheby1LPF(xt, Fs, fparms, dly) <<<<<
    where  yt:  (filtered) output signal, sample rate Fs
           ord: actual filter order
           xt:  input signal, sample rate Fs
           Fs:  sample rate of yt, xt in Hz
           fparams = [N, rp, fL]
           N:   filter order
           rp:  passband ripple in dB
           fL:  cutoff frequency (-3dB) in Hz
           fparams = [fp, fs, gp, gs]
           fp:  passband frequency in Hz
           fs:  stopband frequency in Hz
           gp:  max passband loss in dB
           gs:  min stopband attenuation in dB
           dly: filter delay compensation in sec
    """
    if len(fparms)==4:
        N, wn = ss.cheb1ord(2*fparms[0]/float(Fs),2*fparms[1]/float(Fs),fparms[2],fparms[3])
        rp = fparms[2]
    else:
        N, rp, wn = fparms[0], fparms[1], 2*fparms[2]/float(Fs)
    sos = ss.cheby1(N, rp, wn, output='sos')
    fdly = round(dly*Fs)
    yt = ss.sosfilt(sos, np.hstack((xt, np.zeros(fdly))))
    yt = yt[fdly:]    # filter delay compensation
    return yt, N


def multipath(xt, Fs, tau_n, a_n):
    """
    Multipath channel filter (real-valued)
    >>>>> yt, ord = multipath(xt, Fs, tau_n, a_n) <<<<<
    where  yt:  real-valued output signal, rate Fs
           ord: filter order
           xt:  real-valued input signal, rate Fs
           Fs:  sampling rate in Hz
           tau_n: multipath delays in seconds
           a_n: real-valued attenuation factors at tau_n
    """
    tau_max = np.max(tau_n)
    ix_max = int(np.round(Fs*tau_max))
    ht = np.zeros(ix_max+1)
    ix_tau = np.array(np.round(Fs*np.array(tau_n)),int)
    ht[ix_tau] = a_n
    yt = np.convolve(xt, ht, mode="same")/float(Fs)
    return yt, ix_max+1
    
    
       