# Publications and reports

*Last updated: March 2026*

This page collects peer‑reviewed papers, theses and technical reports that
are directly related to the TRASGO/miniTRASGO detector family and the
analysis methods developed in this repository.  It is intended to serve as a
centralised reference for collaborators preparing new manuscripts or
reporting results; if you publish a new work please add it here.

## Citation guidance

When referring to the hardware or software described in this documentation,
please cite one of the flagship miniTRASGO papers (2025) listed below and, if
appropriate, the digital‑twin or dictionary method publications once they are
published.  A generic BibTeX entry for the NIM A paper is:

```bibtex
@article{soneira2025nima,
  title        = {miniTRASGO: A compact RPC tracker for cosmic ray studies},
  author       = {Soneira-Landin, C. and Blanco, A. and Fraile, L.M. and
                  Garz{\'on}, J.A. and Kornakov, G. and Lopes, L. and
                  Nouvilas, V.M. and Ud{\'\i}as, J.M.},
  journal      = {Nuclear Instruments and Methods in Physics Research A},
  year         = {2025},
  doi          = {10.1016/j.nima.2025.170511},
}
```

Feel free to adapt the author list to match the subset of collaborators
involved.  For internal reports or theses, use the appropriate institutional
identifier (e.g. [UMI handle](http://hdl.handle.net)).

---

## Core miniTRASGO publications (2025)

| Year | Topic | Citation |
|------|-------|----------|
| 2025 | Hardware & detector description | Soneira-Landin *et al.*, *NIM A* (2025). [DOI](https://doi.org/10.1016/j.nima.2025.170511) |
| 2025 | Design and first monitoring results | Soneira-Landin *et al.*, *Adv. Space Res.* (2025). [DOI](https://doi.org/10.1016/j.asr.2025.07.096) |
| 2025 | Conference proceedings (ICRC2025) | Soneira-Landin *et al.* PoS(ICRC2025)1368 (submitted) |

The first paper contains a complete description of the RPC geometry,
electronics and assembly procedures; the second focuses on the network of
stations and initial cosmic‑ray rate measurements.

---

## Ancillary journal articles and conference papers

### Analysis & environmental effects

* Ri{\'a}digos, I., González-Díaz, D. & Pérez-Muñuzuri, V. (2022). Revisiting
the limits of atmospheric temperature retrieval from cosmic-ray
measurements. *Earth and Space Science*, 9, e2021EA001982.
* Ri{\'a}digos *et al.* (2020). Atmospheric temperature effect in secondary
cosmic rays observed with a 2 m² ground-based tRPC detector. *Earth and Space
Science*, 7, e2020EA001131.

### TRASGO / TRAGALDABAS legacy work

Many of the early publications describe the predecessor detectors that
informed miniTRASGO.  Key references include the TRAGALDABAS performance
papers (2014–2017) and the TRASGO proposal (2012).  See the original list
below for full details.

* García-Castro *et al.* (2021). The TRASGO Project – status and results.
  *Phys. Atom. Nucl.*, 84, 1070–1079.
* Saraiva *et al.* (2020). The TRISTAN detector latitude survey. *JINST* 15,
  C09024.
* Assis *et al.* (2018). Autonomous RPCs for a ground array. ICRC2017.
* Garzón *et al.* (2017). TRAGALDABAS first results (arXiv:1701.07277).
* Blanco *et al.* (2015, 2014). TRAGALDABAS detector design (JPhysConfSer, JINST).
* Belver *et al.* (2012). Analysis of cosmic-ray air showers with HADES RPC
  wall. *JINST* 7, P10007.
* Belver *et al.* (2012). RPC HADES-TOF wall cosmic ray test. *NIM A* 661, S114.
* Belver *et al.* (2012). TRASGO proposal. *NIM A* 661, S163–S167.
* Assis *et al.* (2011). R&D for an autonomous RPC station. ICRC2011.

See the full chronological list at the bottom of this page for additional
items.

---

## PhD theses and technical reports

* García Castro, D. (2022). *Cosmic Rays' study with a TRASGO detector.*
  [UMI handle](http://hdl.handle.net/10347/29288). Advisor: Juan P. Garzón Heydt.
* Fontenla Barba, Y. (2019). *Studies on the composition and energy of
  secondary cosmic rays with the Tragaldabas detector.* [UMI handle](http://hdl.handle.net/10347/20655).
* Ajoor, M. (2022). *Study of Cosmic Ray data with the TRISTAN and TRAGALDABAS
  detection systems.* [UMI handle](http://hdl.handle.net/10347/28824).
* Cuenca García, J.J. **Simulation and reconstruction algorithms for a
  commercial muon tomography system.** (no handle available).

---

## Industry collaborations and project reports

* Logicmelt (2025). Stratos DS – Stratospheric temperature prediction using
  artificial intelligence. [link](https://logicmelt.com/en/use-cases_eng/stratos-ds-prediction-of-the-stratosphere-temperature/)
* NAC-Intercom (2025). Project STRATOS – Ground Station for Continuous Monitoring
  of the Stratosphere Temperature through Cosmic Ray Directional Flow.
  [link](https://www.nac-inter.com/en/content/30-project-stratos)

---

### How to add new entries

Edit this markdown file and insert a new bullet in the appropriate section.
Keep the format consistent (year, authors, title, journal/conference, DOI or
URL).  If the paper is still in preparation, mark it as such and update the
entry once it is accepted.

If you produce figures or tables from a new paper that should appear on the
public documentation site, add them to the `plot_list.txt` configuration and
run the `update_plots.sh` script to copy them into `docs/assets`.

---

*(End of publications list)*
