Some notes extracted from the mingo legacy documentation:

- The space-charge effect is something to take into account: 20 yr ago they ignored it and calculations were essentially wrong. This effect is just the electric field produced inside the gap by the ions themselves and its change on the nominal electric field introduced between the plates.

- The text part of the output of the `startDAQ` that are of the form `0xc001 32 3/8 54e03` are just confirming the communication with everyone of the channels (therefore there are 32 orders like that).

- The DCS says `Copying from remote location` because it is connecting to itself.

- We saw in Pablo Cabanelas talk at the 3rd TRASGO meeting (June 27th, 2023, Santiago de Compostela) that a 1 cm Pb layer above TRAGALDABAS would stop electrons and hence improving the capabilities of the detector to identify muons. I think in some sense that is not a surprise: if you have a conflict differentiating electrons from muons and you just stop the electrons then it is somehow clear that you will be much more efficient identifying muons, rigth? **Juanjo, seeing Cabanelas work, showed surprise to the fact that there were less electrons when introducing the lead: he just thought that the muons would interact with the lead emiting even more electrons, not less**.

- Anger camera concept: a potential future branch for the TRASGO project.

- 1 muon per million comes from below the detector: it is due to neutrinos interaction from the other side of the Earth. Right now the detector has a time resolution that is in the limit to allow diferentiation between above and below: if we separated just the layers a bit then we could achieve the needed resolution.
