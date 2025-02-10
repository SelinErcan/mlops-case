CREATE TABLE IF NOT EXISTS loan_data (
  person_age FLOAT,
  person_gender VARCHAR(10),
  person_education VARCHAR(50),
  person_income FLOAT,
  person_emp_exp INT,
  person_home_ownership VARCHAR(20),
  loan_amnt FLOAT,
  loan_intent VARCHAR(50),
  loan_int_rate FLOAT,
  loan_percent_income FLOAT,
  cb_person_cred_hist_length FLOAT,
  credit_score INT,
  previous_loan_defaults_on_file VARCHAR(10),
  loan_status INT,
  CONSTRAINT unique_loan_columns UNIQUE (person_age, person_gender, person_education, person_income, 
                                       person_emp_exp, person_home_ownership, loan_amnt, loan_intent, 
                                       loan_int_rate, loan_percent_income, cb_person_cred_hist_length, 
                                       credit_score, previous_loan_defaults_on_file, loan_status)
);

COPY loan_data(person_age, person_gender, person_education, person_income, person_emp_exp, person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file, loan_status)
FROM '/docker-entrypoint-initdb.d/data/loan_data.csv'
DELIMITER ',' 
CSV HEADER;

CREATE TABLE IF NOT EXISTS monitoring (
    person_gender VARCHAR(10),
    person_education VARCHAR(50),
    person_income FLOAT,
    person_emp_exp INT,
    person_home_ownership VARCHAR(20),
    loan_amnt FLOAT,
    loan_intent VARCHAR(50),
    loan_int_rate FLOAT,
    loan_percent_income FLOAT,
    cb_person_cred_hist_length FLOAT,
    credit_score INT,
    previous_loan_defaults_on_file VARCHAR(10),
    prediction INT NOT NULL,
    actual INT NULL,
    CONSTRAINT unique_monitoring_columns UNIQUE (person_gender, person_education, person_income, person_home_ownership, 
            loan_amnt, loan_intent, loan_int_rate, loan_percent_income, cb_person_cred_hist_length, 
            credit_score, previous_loan_defaults_on_file, prediction, actual)
);
