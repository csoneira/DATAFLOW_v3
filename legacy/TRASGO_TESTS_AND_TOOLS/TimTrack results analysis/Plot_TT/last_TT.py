"""
Created on Tue Jan  9 12:42:06 2024

loaded_matrices[event_number] tiene la forma:

s1 tF1 tB1
s2 tF2 tB2
s3 tF3 tB3
s4 tF4 tB4

Ahora bien, está hecho para tener índices de strip.

@author: cayesoneira / UCM
+ j.a. garzon / LabCAF. USC-IGFAE 
"""

def tt_nico(filename):
    Limit = True
    limit_number = 5000
    import numpy as np
    import scipy.linalg as linalg
    import scipy.stats as stats
    from math import sqrt,pi,floor
    import matplotlib.pyplot as plt
    #import time
    #%% A few important functions
    def fmgx(nvar, npar, vs, ss, zi): # G matrix for t measurements in X-axis
        mg = np.zeros([nvar, npar])
        XP = vs[1]; YP = vs[3]; S0 = vs[5]
        kz = sqrt(1 + XP*XP + YP*YP)
        kzi = 1 / kz
        #
        mg[0,2] = 1
        mg[0,3] = zi
        mg[1,1] = kzi * S0 * XP * zi
        mg[1,3] = kzi * S0 * YP * zi
        mg[1,4] = 1
        mg[1,5] = kz * zi
        mg[2,0] = ss
        mg[2,1] = ss * zi
        return mg
    def fmwx(nvar, sy, sts, std): # Weigth matrix 
        mw = np.zeros([nvar, nvar])
        mw[0,0] = 1/(sy*sy)
        mw[1,1] = 1/(sts*sts)
        mw[2,2] = 1/(std*std)
        return mw
    def fvmx(nvar, vs, lenx, ss, zi): # Fitting model array with X-strips
        vm = np.zeros(nvar)
        X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]; S0 = vs[5]
        kz = np.sqrt(1 + XP*XP + YP*YP)
        #
        xi = X0 + XP * zi
        yi = Y0 + YP * zi
        ti = T0 + kz * S0 * zi
        th = 0.5 * lenx * ss   # tau half
        lxmn = -lenx/2
        #
        vm[0] = yi
        vm[1] = th + ti
        vm[2] = ss * (xi-lxmn) - th
        return vm
    def fmkx(nvar, npar, vs, sy, sts, std, ss, zi): # K matrix
        mk  = np.zeros([npar,npar])
        mg  = fmgx(nvar, npar, vs, ss, zi)
        mgt = mg.transpose()
        mw  = fmwx(nvar, sy, sts, std)
        mk  = mgt @ mw @ mg
        return mk
    def fvax(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi): # va vector
        va = np.zeros(npar)
        mw = fmwx(nvar, sy, sts, std)
        vm = fvmx(nvar, vs, lenx, ss, zi)
        mg = fmgx(nvar, npar, vs, ss, zi)
        vg = vm - mg @ vs
        vdmg = vdat - vg
        va = mg.transpose() @ mw @ vdmg
        return va
    def fs2(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi):
        va = np.zeros(npar)
        mk = fmkx(nvar, npar, vs, sy, sts, std, ss, zi)
        va = fvax(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi)
        vm = fvmx(nvar, vs, lenx, ss, zi)
        mg = fmgx(nvar, npar, vs, ss, zi)
        mw = fmwx(nvar, sy, sts, std)
        vg = vm - mg @ vs
        vdmg = vdat - vg
        #
        mg = fmgx(nvar, npar, vs, ss, zi)
        sk = vs.transpose() @ mk @ vs
        sa = vs.transpose() @ va
        s0 = vdmg.transpose() @ mw @ vdmg
        s2 = sk - 2*sa + s0
        return s2
    def fmahd(npar, vin1, vin2, merr): # Mahalanobis distance
        #vdif = np.ones(npar)
        #vdsq = np.ones(npar)
        #vsig = np.ones(npar)
        #
        vdif  = np.subtract(vin1,vin2)
        vdsq  = np.power(vdif,2)
        verr  = np.diag(merr,0)
        vsig  = np.divide(vdsq,verr)
        dist  = np.sqrt(np.sum(vsig))
        return dist
    def fres(vs, vdat, sy, sts, std, lenx, ss, zi):  # Residuals array
        X0 = vs[0]; XP = vs[1]; Y0 = vs[2]; YP = vs[3]; T0 = vs[4]; S0 = vs[5]
        kz = sqrt(1 + XP*XP + YP*YP)
        #  Fitted values
        xf  = X0 + XP * zi
        yf  = Y0 + YP * zi
        tff = T0 + S0 * kz * zi + xf * ss
        tbf = T0 + S0 * kz * zi + (lenx - xf) * ss
        tsf = 0.5 * (tff + tbf)
        tdf = 0.5 * (tff - tbf)
        #  Data
        yd  = vdat[0]
        tfd = vdat[1]
        tbd = vdat[2]
        tsd = 0.5*(tfd + tbd)
        tdd = 0.5*(tfd - tbd)
        # Residuals
        yr   = (yf - yd)/sy
        tsr  = (tsf  - tsd)/sts
        tdr  = (tdf  - tdd)/std
        # Residuals array
        vres = [yr, tsr, tdr]
        return vres
    #%%#   A few constants
    vc  = 300 #mm/ns
    sc  = 1/vc
    vp  = 2/3 * vc  # velocity of the signal in the strip
    ss  = 1/vp
    #
    cocut = 1  # convergence cut
    #%% Detector layout
    nplan = 4
    vz    = [0,100,200,400]
    #zor   = vz[0];
    lenx  = 300
    mystp = [[4, 68, 132, 196],
            [4, 103, 167, 231],
            [4, 68, 132, 196],
            [4, 103, 167, 231]]
    mystp = np.array(mystp) - 150
    mwstp = [[64, 64, 64, 99],   # effective strip width
             [99, 64, 64, 64],
             [64, 64, 64, 99],
             [99, 64, 64, 64]]
    # Tentative uncertainties
    sts = 0.75    # t0 uncertainty (it was 0.3)
    std = 0.05  # tx uncertainty (it was 0.085)
    #
    #%%    
    npar = 6
    nvar = 3
    fdat_q = []
    fdat = []
    vdat = np.zeros(nvar)
    mfit = np.zeros(npar)
    vchi = []   # chi2 values
    vcha = []   # Charges
    mcha = []   # Charges
    vchr = []
    vc2y = []
    vcts = []
    vctd = []
    vcsf = []  # chi2 survival function
    #%%
    with open(filename,'rb') as file:
        while True:
            try:
                matrix = np.load(file)
                fdat_q.append(matrix)
                fdat.append(matrix[:,0:3])
            except ValueError:
                break
    ntrk  = len(fdat)
    #%%
    X0i = 0.0
    XPi = 0.0
    Y0i = 0.0
    YPi = 0.0
    T0i = 0.0
    S0i = sc
    kz  = sqrt(1 + XPi**2 + YPi**2)
    #
    dist = 10
    nft  = 0     # nb. of fitted tracks
    
    if Limit:
        ntrk = limit_number  # ===================================== Nb. of tracks to be fitted
    
    print("-----------------------------")
    print(f"{ntrk} events to be fitted")
    print("-----------------------------")
    
    for it in range(ntrk):
        if it % 1000 == 0: print(f"Trace {it}")
        mdat = fdat[it]
        
        if np.any(mdat < -9000): continue
        
        nfip = 0
        ndat = 0
        vtag = np.zeros(nplan)
        for ip in range(nplan):
            if mdat[ip][0] <= 0: continue
            vtag[ip] = 1
            nfip     = nfip + 1
        if nfip > 1:
            x0  = X0i
            xp  = XPi
            y0  = Y0i
            yp  = YPi
            t0  = T0i
            s0  = S0i
            vs = [x0,xp,y0,yp,t0,s0]
            mk  = np.zeros([npar, npar])
            mki = np.zeros([npar, npar])
            va  = np.zeros(npar)
            istp = 0   # nb. of fitting steps
            #
            while dist>cocut:
                for ip in range(nplan):
                    if(vtag[ip]== 0): continue
                    zi   = vz[ip]
                    #
                    ist = int(mdat[ip][0])
                    tf  = mdat[ip][1]
                    tb  = mdat[ip][2]
                    ts  = 0.5 * (tf+tb)
                    td  = 0.5 * (tf-tb)
                    #
                    wst     = mwstp[ip][ist-1]
                    yst     = mystp[ip][ist-1] + wst/2
                    vdat[0] = yst
                    vdat[1] = ts
                    vdat[2] = td
                    #
                    sy      = wst / np.sqrt(12)
                    #
                    mk = mk + fmkx(nvar, npar, vs, sy, sts, std, ss, zi)
                    va = va + fvax(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi)
                #
                istp = istp + 1
                merr = linalg.inv(mk)     # Error matrix
                vs0 =vs
                #
                vs  = merr @ va          # sEa equation
                #
                dist = fmahd(npar, vs, vs0, merr)
                # print("***** ", istp, dist)
            dist = 10
            nft = nft + 1  # nb. of fitted tracks
        # Chi2 -------------------------------------------------------
        chi2 = fs2(nvar, npar, vs, vdat, sy, sts, std, lenx, ss, zi)
        vchi.append(chi2)
        # Charges --------------------
        all_charges = fdat_q[it][:,3]
        mcha.append(all_charges)
        # charge = np.mean(all_charges)
        # vcha.append(charge)
        #
        chi2r = 0
        chiy  = 0
        chts  = 0
        chtd  = 0
        ndat  = 0
        for ip in range(nplan):
            if vtag[ip]==0: continue
            #print('---', ip)
            
            ndat = ndat + nvar
            ist  = int(mdat[ip][0])-1 # fired strip nunber
            wst  = mwstp[ip][ist]
            sy   = wst / sqrt(12)
            yst  = mystp[ip][ist] + wst/2 # fired strip coordinate
            tf   = mdat[ip][1]
            tb   = mdat[ip][2]
            vdat = [yst, tf, tb]
            # sts = .5
            # std = 2
            zi  = vz[ip]
            # Residuals array (ystrip, tsum, tdif)
            vres  = fres(vs, vdat, sy, sts, std, lenx, ss, zi)
            ch2y  = chiy + vres[0]**2
            c2ts  = chts + vres[1]**2
            c2td  = chtd + vres[2]**2
            chi2r = chi2r + ch2y + c2ts + c2td
            #print('ip, vres: ',ip, ' ', vres)
        ndf  = ndat - npar    # number of degrees of freedom; it was ndat - npar
        
        # print("-------")
        # print("Degrees of freedom", ndf)
        # print("ndat", ndat)
        # print("nvar", nvar)
        # print("npar", npar)
        
        vc2y.append(ch2y)
        vcts.append(c2ts)
        vctd.append(c2td)
        vchr.append(chi2r)
        chsf = stats.chi2.sf(chi2r, ndf) # chi2 goodnes, survival function = 1 - F(chi2, ndf)
        vcsf.append(chsf)
        
        #print(mfit)
        #print('*',vs)
        vs = np.array(vs)
        mfit = np.vstack((mfit,vs))
        # mfit = np.append(mfit,vs)
    #print('---', mfit)
    mfit = np.delete(mfit,0,0)
    #print('===', mfit)
    
    return mfit, vchr, vcsf, mcha

# mfit, vchi = tt_nico()