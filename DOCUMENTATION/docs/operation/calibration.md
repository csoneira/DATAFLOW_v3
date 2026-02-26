# Calibrating
There are some calibration processes that must be done when using the mingo for the first time in a certain location, but also every once in a while, specially before observation campaigns.

## Time to position on the strip calibration
To get the position we need the velocity of propagation of the signal through the strip (and most likely on its edges, but since the strips are small it should be negligible), but also to correct for a time offset created by the different length of the wires that come from the strip to the FEE. There are several ways to calibrate this effect for each strip:
- **Quantile method**. The one it is being used. The middle position between the quantiles .5 and .95 is taken as calibration offset.
- **Regression method**. The intercept of the linear regression between the times in the front and in the back of the strip is taken as offset.


## Gas flow meters
The flow meters, that measure in arbitrary units, have an offset, different per each channel, that has to be taken into account. These are not precission devices, so the values are more to be used as a guideline than as a real reference for calculations. It is important to account, though, if the calibration slopes are different or are the same for the four channels, since they have to be prepared as a leak control.

## Performance plateau
The performance of the RPC is strongly related with the applied HV. As the HV grows, the detector gains in efficiency, but also measures much more streamers (uncontrolled avalanches that deposit a charge sligthly bigger than the usual avalanche). The *sweet spot* is that with the better compromise between streamers (should be around 1-2%), efficiency, charge, rate, etc. **Actually we should decide which are the best indicators of good performance**. In RPC this optimum is usually given by a particular value of the Townsend: the rate between electric field and density of the gas. This means that changing the location, and therefore the pressure, temperature... will change the density of the gas and so the Townsend value.

This analysis has to be performed at every telescope location (and maybe several times a year or at the beginning of a measure campaign) since it depends on temperature, pressure, etc.

There is a script, `home/rpcuser/bin/HV/plateau.sh`, that performs the plateau analysis by scanning in a given range of HV values. This range can be modified, as well as its finesse and duration.

## Time-to-Digital Converters (TDC)
They need calibration: they give three numbers, they need three strings. Alberto Blanco knows more about this, but it requires more subtle work. Some very recognizable errors arise when this calibration is not well performed.

## Charge calibration
The calibration in charge has to be performed for two different components.

### Charge offset
The zero of the charge spectrum (in AU or in C) has to be obtained to eliminate the offset. The algorithm is complex and can be worked on.

### FEE time-to-charge calibration curve
The width of the LVDS is related with the charge of the original signal, but not in a linear way. This means that to obtain a reliable charge spectrum, that will be different in shape in AU and in C, we will need the transformation to Coulombs. All this question comes from the method from which the LVDS width is created.

There are different methods to get a LVDS width from the original RPC signal. For example using the integrated signal: we could measure eventually at a relatively long time the maximum height to know the total charge. This method is slow, since the integrated signal just reaches a top that later has to be lowered. The lowering of the signal is so slow that the dead time is very high.

Other usual technique is the **TOT** (Time over Threshold), which is a method that gets the width of the LVDS according to how much time the charge signal is over a certain discriminator. Since the shape of the charge respect to time is presumably dependent only on the total charge (meaning that total integrated value, amplitude and shape are directly related), we can guess the total height of the signal from the width at a certain height. According to the type of signal to which we are applying the TOT, we can classify:
- Standard: just the regular Q vs. time function.
- Integrated and derivated: the signal is integrated and at the same time it is derivated so it would be overall thicker than the original RPC signal but it is thick enough to allow the TOT to apply. The speed of the derivation (which lowers the signal) determines why it is called fast electronics; the faster the better, even though there is a handicap in lowering the signal to fast: THE same charge avalanche inside the RPC could be measured twice as two different, independent signals, and this is because the ions take microseconds to totally get to the strip: they are much slower than the electronics. This has to be solved if we want faster electronics, but some filters could work.
From the TOT and knowing the method we apply to modify the signal we can eventually derive a value for the charge in AU. If we can get the *charge calibration curve* we can transform from charge in AU (arbitrary units) to those of proper charge: Coulombs. **And the transformation is non-linear, so it will slightly change the spectrum shape**.

Here we include the FEE calibration setup to obtain the time-to-charge calibration curve.

![FEE calibration setup](https://github.com/cayesoneira/miniTRASGO-documentation/assets/93153458/c8b0de84-0890-4c57-9012-c443c591541c)
