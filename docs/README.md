# Documentation

Start with the category index that owns the feature you are working on.
Current implementation contracts, future designs, and historical observations
are deliberately separated.

| Category | Contents |
|---|---|
| [UI](ui/README.md) | Rendering, widgets/input/layout, panels, Console, Observatory, and popup presentation |
| [OSINT](osint/README.md) | Registered tools, jobs/cache, setup, and sensitive-pattern scanning |
| [Desktop input](input/README.md) | Input-tool contracts, grounded targeting, cursor/capture, and OS hit testing |
| [Model and training](model/README.md) | Autograd, loss, parameters, RMSNorm/readout, batching, math, telemetry, plans, and investigations |
| [Legacy](legacy/) | Retired architectures and historical replacement snippets; not current instructions |

## Cross-feature contracts

These remain at the root because they define boundaries shared by multiple
categories:

- [Reasoning data contract](GRIM_REASONING_DATA_CONTRACT.md): model-visible
  structured text, Goal/evidence, supervision, and environment interactions.
- [World Lab](WORLD_LAB.md), [reference](WORLD_LAB_REFERENCE.md), and
  [Live World](LIVE_WORLD.md): environment/resource behavior and UI.
- [Plugin API](PLUGIN_API.md) and [retained handlers](RETIRED_AGENT_TOOLS.md):
  registered capabilities versus retained host services.
- [Configuration](CONFIG_SYSTEM.md), [voice](VOICE_TUNING_GUIDE.md), and
  [physical perception](PHYSICAL_PERCEPTION_SUBSYSTEM.md).
- [Log attribution](LOG_FILE_CONVENTION.md): identify exact run/log sources.
- [Plateau investigation](PLATEAU_BUG_INVESTIGATION.md): historical run evidence,
  not current architecture. Its established root reference is retained.

## Historical project reports

- [April 3-9, 2026](reports/Weekly_Report_April_3-9_2026.md)
- [April 10-16, 2026](reports/Weekly_Report_April_10-16_2026.md)

Reports cover the wider project, so they do not belong exclusively to model
documentation. Their current-code reconciliation is separate from original
reporting-period measurements.

Use existing guides for new feature knowledge instead of creating competing
quick references, implementation summaries, or one-off fix recipes. Link to
the actual source and distinguish source inspection from executed validation.
