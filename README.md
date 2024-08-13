## Restructure Project
Restructure codebase into modularized project structure that the usecase can be deployed as a microservice.
```bash
├── src
│   ├── main           # main logic for training, predicting and visualisation.
│   └── utils          # helper functions.
├── models             # storage of resources such as trained models.
├── tests              # contains projects testing suite.
├── config.yml         # stores project configurations in json format.
├── app.py             # fastapi application logic.
├── docker-compose.yml # Docker configurations.
├── Dockerfile         # machine instructions to setup the application and run inside Docker as a micro-service.
├── README.md
├── requirements.txt   # Python dependancies for installation with pip.
├── run_app.py         
└── run.py             # entry point of the project for local usage
```
## Refactoring Code 
Start refactoring code from jupyter notebook to the targeting scripts. Using following practices in code refactoring: 

- Use configuration files for all configuration and variable parameters.
- Pull all required dependencies in a separate requirement.txt file (already provided)
- Rewrite training, prediction, visualization code cells into modularized functions with docstrings and documentations, enable logging for each modularized functionalities.
- Write unit tests for functions to ensure the functionalities (Currently missing in my solution)
- Apply linting / code formating with pre-commit hooks
- Containerize the app to ensure it can be reproduciable and running in isolated environments.
- Documentation and readme.

## Test productionized service locally

### Set up
- Prerequisites: python >= 3.9 installed
- Docker installed
### Deployment the docker service
```bash
docker-compose up --build
```
### Swagger UI Testing
Open any web browser and navigate to http://localhost:8000/docs to view the Swagger UI documentation and test the API endpoints.

**For Post Methods /prepare-data/, /train/, /predict/, test with config_path like following**
```bash
# Request body
{
  "config_path": "config.yml"
}
```
**For Get Methods /plot-dataset/, /plot-classifier-output/, /plot-inference-output/, test with config_path like following**

![api_test_example]("Diagrams/API_call.png")
### Shut down the service
```bash
docker-compose down
```
## Deploy in Cloud Environment

Here I proposed two architectures (AWS and Azure) of how to deploy the microservice respectively using AWS SageMaker and Azure ML Studio. Both provide managed MLflow server for experiments tracking and logging, also Model Registry for model control. 

### Azure MLOps Deployment
![Azure MLOps Architecture]("/Diagrams/Azure_MLOps.png")

In the Azure MLOps architecture utilizes separate DEV and PRD resource groups, with Blob Storage and MLflow for managing the ML lifecycle. In DEV, data validation, model training, and evaluation are streamlined with Azure ML Studio built-in MLOps features, while in PRD, models are monitored and deployed with alert systems in place. Using Azure DevOps CI/CD pipelines to automate resource provisioning, code quality checks, and deployments, with a shared Feature Store and Model Registry bridging both environments

![AWS MLOPs Architecture]("./Diagrams/AWS_MLOps.png")

In the AWS MLOps architecture integrates GitLab CI/CD pipelines with AWS services like SageMaker and S3, orchestrated by MLflow for managing the entire ML lifecycle. Data flows from external sources into S3 via a DB connector, undergoes preprocessing and quality checks, and then moves to SageMaker for training. Post-training, models are evaluated, registered, and upon approval, deployed using deployment containers. An API layer facilitates interaction and reporting for different roles, ensuring seamless end-to-end MLOps with clear separation between development and production environments.

## Roadmap
- For the modularized code it self, we need to include MLflow logging and experiment tracking logic.
- Unit testing is still missing. 
- Infrastructure wise the IaC part is missing and needs to be implement for resources management. 
