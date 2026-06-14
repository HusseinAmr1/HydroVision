# Flood Project Architecture Diagrams

This document reflects the current project structure and runtime behavior of the repository.

The diagrams below are intentionally split by concern:
- overall system architecture
- web portal runtime
- prediction request flow
- training pipeline
- artifact and model lifecycle

The current production path is `Pipeline V3 only`. Legacy training/runtime is no longer part of the active flow.

## 1. Overall System Architecture

```mermaid
flowchart LR
    U1[End User]
    U2[Operator / Developer]

    subgraph Web["Web Portal"]
        WF[web_frontend/index.html + app.js + styles.css]
        WA[web_api.py FastAPI]
        DB[(web_data.db SQLite)]
    end

    subgraph Streamlit["Desktop Dashboard"]
        SG[run_gui.py]
        SA[app.py Streamlit]
    end

    subgraph Core["Project Core"]
        PJ[project.py Orchestrator]
        UP[segmentation_pipeline.py DL Backend]
        MS[model_security.py]
        EU[env_utils.py]
    end

    subgraph Models["Active Runtime Artifacts"]
        OD[outputs_smp_deeplabv3plus_resnet50_livefix_restart_20260310_181858]
        S1[pipeline_v3_model_s1.pth]
        S2[pipeline_v3_model_s2.pth]
        RW[risk_model_with_weather_s1_pipeline_v3.joblib]
        RN[risk_model_no_weather_global_pipeline_v3.joblib]
        RT[risk_model_temporal_gb_s1_pipeline_v3.joblib]
        REG[model_registry.json]
        PROF[input_profile.json]
    end

    subgraph Data["Local Data"]
        DS[Dataset S1 / S2 images]
        WX[Final_Full_Data_Matched.csv]
        ERA[ERA5_Final1_Combined.csv]
    end

    subgraph External["External Services"]
        GG[Google Sign-In]
        GM[Gemini API]
        LS[Copernicus Data Space / Sentinel APIs]
    end

    U1 --> WF
    WF --> WA
    WA --> DB
    WA --> PJ
    WA --> MS
    WA --> GG
    WA --> GM
    WA -. experimental .-> LS

    U2 --> SG
    SG --> SA
    SA --> PJ
    SA --> UP

    PJ <--> UP
    PJ --> DS
    PJ --> WX
    PJ --> ERA

    WA --> OD
    SA --> OD
    UP --> OD
    OD --> S1
    OD --> S2
    OD --> RW
    OD --> RN
    OD --> RT
    OD --> REG
    OD --> PROF

    EU --> WA
    EU --> SA
    EU --> SG
```

## 2. Web Portal Runtime

```mermaid
flowchart TD
    A[start_web.bat] --> B[run_web.py]
    B --> C[load .env]
    C --> D[dependency check]
    D --> E[uvicorn web_api:app]

    E --> F[/web static frontend]
    E --> G[/api/auth/*]
    E --> H[/api/predict]
    E --> I[/api/predictions/*]
    E --> J[/api/chat]
    E --> K[/api/live-satellite/*]

    subgraph WebAPI["web_api.py"]
        L[load models into LoadedModels]
        M[load auth + JWT + roles]
        N[load DB session]
        O[load assistant mode]
    end

    E --> L
    E --> M
    E --> N
    E --> O

    subgraph Frontend["web_frontend"]
        P[React UI]
        Q[Prediction form]
        R[History panel]
        S[Latest result]
        T[AI assistant panel]
    end

    F --> P
    P --> Q
    P --> R
    P --> S
    P --> T
```

## 3. Web Prediction Request Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as React Frontend
    participant API as web_api.py
    participant Core as project.py
    participant DL as segmentation_pipeline.py
    participant Art as outputs runtime dir
    participant DB as web_data.db
    participant AI as Gemini API

    User->>UI: Upload TIFF + Run Detection & Forecast
    UI->>API: POST /api/predict (multipart + bearer token)
    API->>API: validate auth, rate limit, TIFF type, file size
    API->>API: reject likely label-mask uploads
    API->>API: read TIFF + infer sensor + inspect geo metadata
    API->>Core: merge weather from CSV / ERA5 anchor
    API->>DL: predict_pipeline_mask_auto(...)
    DL-->>API: pred_mask + pred_prob + infer_meta
    API->>Core: summarize_prediction_features(...)
    API->>Core: route_risk_prediction(...)

    alt detection_label == 1
        API->>API: suppress temporal forecast
        API->>Core: build payload as detection-first result
    else no flood detected
        API->>Core: predict_temporal_risk(...)
        API->>Core: build_prediction_analysis(...)
    end

    API->>API: save_artifacts(...)
    API->>Art: preview PNG + JSON + preview metadata
    API->>DB: save PredictionRecord
    API-->>UI: prediction payload + preview_url

    UI->>API: GET /api/predictions/{id}/preview/{name}
    API-->>UI: preview image (auth-protected)

    User->>UI: Ask assistant about result
    UI->>API: POST /api/chat with prediction_id
    API->>DB: load latest prediction context
    API->>AI: Gemini chat request
    AI-->>API: assistant reply
    API-->>UI: reply + reply_source
```

## 4. Training Pipeline

```mermaid
flowchart TD
    A[start_train_pipeline.bat] --> B[project.py train-pipeline]
    B --> C[run_train_pipeline_command]
    C --> D[maybe_launch_train_live_monitor]
    C --> E[segmentation_pipeline.train_pipeline_models]

    E --> F[discover dataset pairs]
    F --> G[export dataset metadata]

    subgraph S1S2["Per Sensor Training"]
        H[split train / val]
        I[compute normalization stats]
        J[build train patches]
        K[build val patches]
        L[build segmentation model]
        M[train epochs]
        N[patch metrics + image metrics]
        O[threshold tuning]
        P[save best checkpoint]
        Q[save preview + reports]
    end

    G --> H
    Q --> R[global segmentation metrics]
    R --> S[train risk with weather model]
    S --> T[train risk no-weather model]
    T --> U[train temporal risk model]
    U --> V[write model_registry.json]
    V --> W[write input_profile.json]
    W --> X[write pipeline_v3_train_report.json]
    X --> Y[update active_backend.json]
```

## 5. Artifact and Runtime Lifecycle

```mermaid
flowchart LR
    A[train_pipeline_models] --> B[output run directory]

    subgraph RunDir["Active output dir"]
        C[pipeline_v3_model_s1.pth]
        D[pipeline_v3_model_s2.pth]
        E[risk_model_with_weather_s1_pipeline_v3.joblib]
        F[risk_model_no_weather_global_pipeline_v3.joblib]
        G[risk_model_temporal_gb_s1_pipeline_v3.joblib]
        H[model_registry.json]
        I[input_profile.json]
        J[pipeline_v3_train_report.json]
        K[pipeline_v3_val_metrics_global.json]
        L[pipeline_v3_training_progress.png]
    end

    B --> C
    B --> D
    B --> E
    B --> F
    B --> G
    B --> H
    B --> I
    B --> J
    B --> K
    B --> L

    subgraph Consumers["Consumers"]
        M[web_api.py]
        N[app.py Streamlit]
        O[run_gui.py preflight]
        P[project.py predict]
    end

    C --> M
    D --> M
    E --> M
    F --> M
    G --> M
    H --> M
    I --> M

    C --> N
    D --> N
    E --> N
    F --> N
    G --> N
    H --> N
    I --> N

    B --> O
    B --> P
```

## 6. Auth and Assistant Flow

```mermaid
flowchart LR
    U[Browser User] --> W[React Web UI]

    W --> A[/api/auth/register]
    W --> B[/api/auth/login]
    W --> C[/api/auth/google]
    C --> G[Google Identity Services]

    A --> API[web_api.py]
    B --> API
    C --> API
    API --> DB[(web_data.db users)]
    API --> JWT[JWT bearer token]

    W --> D[/api/chat]
    D --> API
    API --> HX[(chat_messages table)]
    API --> PR[(predictions table)]

    alt Gemini configured
        API --> GM[Gemini API]
        GM --> API
    else fallback
        API --> LF[local project-aware fallback]
    end

    API --> W
```

## 7. Live Satellite Experimental Path

```mermaid
flowchart TD
    U[User] --> W[Web Portal]
    W --> A[/api/live-satellite/search]
    W --> B[/api/live-satellite/predict]

    A --> LS[Copernicus Data Space search]
    B --> LS2[Copernicus chip fetch]
    LS2 --> API[web_api.py]
    API --> CORE[project.py]
    API --> DL[segmentation_pipeline.py]
    API --> DB[(web_data.db)]
    API --> OUT[preview + prediction artifacts]
```

## 8. Suggested Usage

If you need one diagram only for a report or presentation, use:
- Diagram 1 for the full project overview

If you need a technical defense, use:
- Diagram 1
- Diagram 3
- Diagram 4
- Diagram 5

If you need the web portal explanation only, use:
- Diagram 2
- Diagram 3
- Diagram 6
- Diagram 7
