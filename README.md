# Li-ion Digital Shadow Dashboard

This project now includes a Streamlit dashboard for browsing the Li-ion battery digital shadow.

## Prepare dashboard artifacts

Run:

```bash
python scripts/prepare_dashboard_data.py --mat-dir mat_files --output-dir artifacts
```

This precomputes and writes:

- global summary Parquet files
- per-battery sample Parquet partitions
- per-battery sample-shadow Parquet partitions
- ECM metrics and parameter JSON files
- a manifest used by the dashboard loaders

## Launch the dashboard

Run:

```bash
streamlit run app.py
```

## Fetch model

- Global summary data is loaded eagerly at app startup.
- Battery detail traces are loaded lazily after a battery and cycle range are selected.
- Raw MAT parsing is kept out of the interactive dashboard path.
