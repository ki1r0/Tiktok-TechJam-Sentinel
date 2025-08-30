# PII Detection Model Comparison & Deployment

## 🏆 Model Performance Summary (80-20 Split Validation)

| Model               | Architecture   | Parameters | Validation F1-Macro | Performance Rank |
| ------------------- | -------------- | ---------- | ------------------- | ---------------- |
| **Ettin 1B**        | Custom Encoder | 1B         | **0.9883**          | 🥇 1st           |
| **Ettin 400M**      | Custom Encoder | 400M       | **0.9856**          | 🥈 2nd           |
| **DeBERTa v3 Base** | DeBERTa-v3     | 184M       | **0.9824**          | 🥉 3rd           |
| **ModernBERT Base** | ModernBERT     | 184M       | **0.9736**          | 4th              |

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Train Ettin 1B Model (Best Performance)

```bash
python3 train-ettin.py
```

### Train Deberta Base Model

```bash
python3 train-deberta.py
```

### Train ModernBERT Base Model

```bash
python3 train-modernbert.py
```

### Run evaluation in real simulation case

```bash
python3 eval.py
```

## 📊 Real-Time Simulation Examples

Below are examples showing the real-time PII detection capabilities of each model:

### Example 1: Personal Information

**Input**: "Hi, my name is John Smith and I live at 123 Main Street, New York. You can reach me at john.smith@email.com or call (555) 123-4567."

#### Ettin 1B Results

```
🔍 LASTNAME: 'Smith' (conf: 0.969)
🔍 BUILDINGNUMBER: '123' (conf: 1.000)
🔍 STREET: 'Main' (conf: 1.000)
🔍 STREET: 'Street,' (conf: 1.000)
🔍 STATE: 'New' (conf: 0.999)
🔍 STATE: 'York.' (conf: 0.999)
🔍 USERNAME: 'john.smith@email.com' (conf: 0.852)
🔍 PHONENUMBER: '(555)' (conf: 1.000)
🔍 PHONENUMBER: '123-4567.' (conf: 1.000)
```

#### ModernBERT Base Results

```
🔍 FIRSTNAME: 'John' (conf: 0.998)
🔍 LASTNAME: 'Smith' (conf: 0.981)
🔍 BUILDINGNUMBER: '123' (conf: 0.999)
🔍 STREET: 'Main' (conf: 0.999)
🔍 STREET: 'Street,' (conf: 0.999)
🔍 STATE: 'New' (conf: 0.981)
🔍 EMAIL: 'john.smith@email.com' (conf: 1.000)
🔍 PHONENUMBER: '(555)' (conf: 1.000)
🔍 PHONENUMBER: '123-4567.' (conf: 0.984)
```

### Example 2: Financial Information

**Input**: "Please charge my credit card 4532-1234-5678-9012, the CVV is 123. My account number is AC789012345 and routing number is 021000021."

#### Ettin 1B Results

```
🔍 CREDITCARDNUMBER: '4532-1234-5678-9012,' (conf: 0.898)
🔍 CREDITCARDCVV: '123.' (conf: 1.000)
🔍 BIC: 'AC789012345' (conf: 0.868)
🔍 ACCOUNTNUMBER: '021000021.' (conf: 1.000)
```

#### ModernBERT Base Results

```
🔍 CREDITCARDNUMBER: '4532-1234-5678-9012,' (conf: 0.996)
🔍 CREDITCARDCVV: '123.' (conf: 0.998)
🔍 ZIPCODE: '021000021.' (conf: 0.996)
```

### Example 3: Government IDs

**Input**: "My SSN is 123-45-6789 and my driver's license number is DL987654321. I was born on 03/15/1985 and I'm 38 years old."

#### Ettin 1B Results

```
🔍 SSN: '123-45-6789' (conf: 1.000)
🔍 VEHICLEVRM: 'DL987654321.' (conf: 0.855)
🔍 DOB: '03/15/1985' (conf: 1.000)
🔍 AGE: '38' (conf: 1.000)
🔍 AGE: 'years' (conf: 0.881)
```

#### ModernBERT Base Results

```
🔍 ACCOUNTNUMBER: '123-45-6789' (conf: 0.987)
🔍 ZIPCODE: 'DL987654321.' (conf: 0.651)
🔍 DOB: '03/15/1985' (conf: 0.997)
🔍 AGE: '38' (conf: 0.945)
🔍 AGE: 'years' (conf: 0.945)
```

## 🎯 Key Observations

### Detection Capabilities

- **58 PII Types** supported including:
  - Personal: FIRSTNAME, LASTNAME, EMAIL, PHONENUMBER, SSN, DOB, AGE
  - Financial: CREDITCARDNUMBER, CREDITCARDCVV, ACCOUNTNUMBER, IBAN, BIC
  - Location: ADDRESS, CITY, STATE, ZIPCODE, BUILDINGNUMBER
  - Technical: IP, MAC, USERNAME, PASSWORD, URL
  - Government: SSN, VEHICLEVIN, VEHICLEVRM

### Confidence Scores

- Models provide confidence scores (0.0-1.0) for each detected PII
- Higher confidence indicates more reliable detection
- Threshold-based filtering available (default: 0.5)
