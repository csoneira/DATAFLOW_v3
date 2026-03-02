# Collaborators

This section summarizes the team structure and primary contacts for detector operations, software development, and analysis coordination.

## Collaboration scope

The project combines hardware deployment, station operations, and software development across multiple institutions.

### Core institutions

- Universidad Complutense de Madrid (UCM), Spain
- Universidade de Santiago de Compostela (USC), Spain
- LIP Coimbra, Portugal
- IFIC Valencia, Spain
- Warsaw University of Technology, Poland
- Benemerita Universidad Autonoma de Puebla, Mexico
- Universidad Pedagogica y Tecnologica de Colombia, Colombia

![Collaboration network map (repository figure)](/assets/repository_figures/network_map_attic.png)

Source path:
- `.ATTIC/network_map.png`

## Collaboration leadership and responsibilities

| Responsibility | Person | Scope |
| --- | --- | --- |
| Concept lead | Juan A. Garzon | Project concept and scientific framing |
| Detector construction and maintenance lead | Alberto Blanco | Detector build, hardware maintenance, and detector operations readiness |
| Analysis software lead and Madrid station responsible | C. Soneira-Landin | Analysis mother code (`MASTER`), software governance, and Madrid station software responsibility |
| Warsaw station responsible (collaborator) | Georgy Kornakov | Warsaw station technical collaboration and station-specific coordination |
| Monterrey station responsible | Humberto | Monterrey station coordination |
| Puebla station responsible | Oliver | Puebla station coordination |

For direct contact details, see [Appendix Contact List](../appendices/contact-list.md).

## Documentation ownership model

- Hardware procedures: maintained in [Hardware](../hardware/index.md).
- Software architecture and workflows: maintained in [Software](../software/index.md).
- Operational runbooks and incidents: maintained in [Operational Notes](../operations/index.md).
- Standards and reproducibility rules: maintained in [Conventions and Standards](../standards/index.md).

## Collaboration practice

- Use pull requests for any procedural or behavior-impacting documentation update.
- Keep incident notes date-stamped and reproducible.
- When code behavior changes, update both technical docs and runbooks in the same change set.
