
# Loan Approval Dataset MLOps Case

## 1) Run Docker and PostgreSQL

### Build and Run Docker:
```bash
docker-compose up --build
```

### Other Docker Commands:
- **Stop and remove any existing containers:**
  ```bash
  docker-compose down
  ```

- **Start the services in the background:**
  ```bash
  docker-compose up -d
  ```

- **Check the logs of the PostgreSQL container:**
  ```bash
  docker logs postgres-container
  ```

### Access PostgreSQL to Check Data:
- Get inside the PostgreSQL container:
  ```bash
  docker exec -it postgres-container bash
  ```

- Connect to the PostgreSQL database:
  ```bash
  psql -U myuser -d loan_database -h localhost -p 5432
  ```

- Execute the following query to check data:
  ```sql
  SELECT * FROM loan_data LIMIT 10;
  ```

---

## 2) Run EDA Notebook for Initial Training and Save Model

Run the Exploratory Data Analysis (EDA) notebook to perform initial training and save the model.

---

## 3) Predict Data

To predict data, use the following `curl` command to upload the test data:
```bash
curl -X POST -F "file=@data/test_data.csv" http://localhost:8000/predict
```

---

## 4) Access Prometheus and Grafana UI on Local Machine

- **Prometheus:** [http://localhost:9090/](http://localhost:9090/)
- **Grafana:** [http://localhost:3000/](http://localhost:3000/)
