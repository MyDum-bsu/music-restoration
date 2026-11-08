# Source discovery and lossless provenance

Search the recording and edition, then the file format. Capture artist/title, duration, album, release year, edition/catalog number, ISRC when available, and source URL. Confirm studio/live/edit/remix/remaster explicitly. ISRC is supporting evidence for recording identity, not proof of identical mastering; identical recording IDs can occur on several releases. Compare audio fingerprints/duration and inspect the edition too.

## Catalog routes

| Route | Use | Completion evidence |
|---|---|---|
| Existing local collection, original downloads, owned CD rips | First lookup; avoid redundant downloads | exact file, hash, edition/rip log where available |
| RuTracker through tracker skill | CD/WEB editions and alternative restorations | topic + downloaded selected files + decode/QC; inspect enough results to cover releases, not just first hits |
| Artist/label Bandcamp | Independent artists and label catalogs | purchase/free-download file in FLAC/ALAC/WAV, actual version |
| Artist/label direct store | Recordings absent from aggregators | explicit lossless download specification and acquired file |
| Qobuz download store | Catalog albums and tracks | acquired FLAC/WAV/ALAC; store availability/account territory checked |
| Presto Music | Classical/jazz catalog | acquired CD-quality/hi-res lossless download |
| Physical CD | When an appropriate downloadable edition cannot be found | secure extraction, log and AccurateRip/CTDB verification where available |

As checked on 2026-11-08, Bandcamp documents FLAC/ALAC/WAV purchases; Qobuz documents FLAC downloads and availability in 27 countries (Belarus absent from the published account-country list); Presto documents FLAC and optional ALAC/WAV. Check current product and territory before promising availability. Catalog discovery does not authorize purchases. A streaming subscription/offline cache does not by itself provide an exportable file for this pipeline.

## Acceptance and ranking

Prefer an authentic, intact edition with good mastering and exact recording identity. CD-quality PCM is sufficient to qualify as lossless; greater bit depth/sample rate is not evidence of better mastering. Vinyl/DSD transfers may be valid alternatives with different noise, speed and mastering; preserve their source label rather than silently substituting them for a digital edition.

Run strict full decode and compare decoded sample count/duration against the container. Preserve acquisition URL/topic, codec, rate, bit depth, SHA-256, edition and rip verification in a sidecar. Spectrum, effective bit depth and codec-like band limits can flag suspicion; none independently prove ancestry. Keep uncertain provenance as unverified. A low-pass master is possible without prior lossy encoding.

Use explicit states: lossless_provenance_supported, lossy_derived, unverified, corrupt, wrong_recording, unavailable, purchase_required. A FLAC export from MP3 stays lossy_derived; processing cannot change it to lossless_provenance_supported. Keep restoration outputs distinct from unprocessed sources.

For an upgrade backlog, keep one row per recording with the current path/provenance and candidate edition URLs, availability, access requirements, verification result and rejection reason. Stop a route after recorded authentication/availability failure and try another catalog; avoid repeatedly scraping a challenge page. Preserve the existing playable file until the replacement passes checks.

## Primary documentation

- [Bandcamp download formats](https://get.bandcamp.help/en/articles/15263234-in-which-formats-can-i-download-my-purchases)
- [Qobuz downloads](https://help.qobuz.com/en/articles/10168-how-to-download-albums-tracks-collections-or-complete-works)
- [Qobuz territories](https://help.qobuz.com/en/articles/10128-where-is-qobuz-available)
- [Presto audio downloads](https://www.prestomusic.com/help/79000083897)
- [MusicBrainz ISRC](https://musicbrainz.org/doc/ISRC)
