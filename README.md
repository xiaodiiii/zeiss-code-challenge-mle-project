** Step 1 **
Refactoring into modularized structure that the usecase can be deployed as a microservice.
```bash
├── src
│   ├── conf           # stores project configurations is json format.
│   ├── main           # main logic for training, predicting and visualisation.
│   ├── resources      # storage of resources such as trained models.
│   ├── template_app   # contains all logic for the flask application.
│   └── utils          # helper functions.
├── tests              # contains projects testing suite.
├── docker-compose.yml # Docker configurations.
├── Dockerfile         # machine instructions to setup the application and run inside Docker as a micro-service.
├── logs.log           # log files storage.
├── READ.md
├── requirements.txt   # Python dependancies for installation with pip.
├── run_app.py         # entry point of the project for the Flask application.
└── run.py             # entry point of the project for local usage
```
