BEST Validation Score:
Ettin 1b: Best F1-Macro Score: 0.9883
Ettin 400m: Best F1-Macro Score: 0.9856
Deberta v3 base: Best F1-Macro Score: 0.9824
Modernbert v3 base: Best F1-Macro Score: 0.9736

Question 1:
Hi, my name is John Smith and I live at 123 Main Street, New York. You can reach me at john.smith@email.com or call (555) 123-4567.

Ettin 1b:
WORD-LEVEL PREDICTIONS:
🔍 LASTNAME: 'Smith' (conf: 0.969)
🔍 BUILDINGNUMBER: '123' (conf: 1.000)
🔍 STREET: 'Main' (conf: 1.000)
🔍 STREET: 'Street,' (conf: 1.000)
🔍 STATE: 'New' (conf: 0.999)
🔍 STATE: 'York.' (conf: 0.999)
🔍 USERNAME: 'john.smith@email.com' (conf: 0.852)
🔍 PHONENUMBER: '(555)' (conf: 1.000)
🔍 PHONENUMBER: '123-4567.' (conf: 1.000)

Ettin 400m:
WORD-LEVEL PREDICTIONS:
🔍 LASTNAME: 'Smith' (conf: 0.760)
🔍 BUILDINGNUMBER: '123' (conf: 1.000)
🔍 STREET: 'Main' (conf: 1.000)
🔍 STREET: 'Street,' (conf: 1.000)
🔍 STATE: 'New' (conf: 1.000)
🔍 STATE: 'York.' (conf: 1.000)
🔍 PHONENUMBER: '(555)' (conf: 0.999)
🔍 PHONENUMBER: '123-4567.' (conf: 1.000)

Deberta v3 base:
WORD-LEVEL PREDICTIONS:
🔍 FIRSTNAME: 'John' (conf: 0.903)
🔍 LASTNAME: 'Smith' (conf: 0.841)
🔍 BUILDINGNUMBER: '123' (conf: 1.000)
🔍 STREET: 'Main' (conf: 1.000)
🔍 STREET: 'Street,' (conf: 1.000)
🔍 STATE: 'New' (conf: 0.999)
🔍 STATE: 'York.' (conf: 1.000)
🔍 PHONENUMBER: '(555)' (conf: 1.000)
🔍 PHONENUMBER: '123-4567.' (conf: 1.000)

modernbert base:
WORD-LEVEL PREDICTIONS:
🔍 FIRSTNAME: 'John' (conf: 0.998)
🔍 LASTNAME: 'Smith' (conf: 0.981)
🔍 BUILDINGNUMBER: '123' (conf: 0.999)
🔍 STREET: 'Main' (conf: 0.999)
🔍 STREET: 'Street,' (conf: 0.999)
🔍 STATE: 'New' (conf: 0.981)
🔍 EMAIL: 'john.smith@email.com' (conf: 1.000)
🔍 PHONENUMBER: '(555)' (conf: 1.000)
🔍 PHONENUMBER: '123-4567.' (conf: 0.984)

Question 2:
Please charge my credit card 4532-1234-5678-9012, the CVV is 123. My account number is AC789012345 and routing number is 021000021.

Ettin 1b:
WORD-LEVEL PREDICTIONS:
🔍 CREDITCARDNUMBER: '4532-1234-5678-9012,' (conf: 0.898)
🔍 CREDITCARDCVV: '123.' (conf: 1.000)
🔍 BIC: 'AC789012345' (conf: 0.868)
🔍 ACCOUNTNUMBER: '021000021.' (conf: 1.000)

Ettin 400m:
WORD-LEVEL PREDICTIONS:
🔍 CREDITCARDNUMBER: '4532-1234-5678-9012,' (conf: 0.803)
🔍 CREDITCARDCVV: '123.' (conf: 1.000)
🔍 VEHICLEVIN: 'AC789012345' (conf: 0.601)
🔍 ACCOUNTNUMBER: '021000021.' (conf: 0.991)

Deberta v3 base:
WORD-LEVEL PREDICTIONS:
🔍 CREDITCARDNUMBER: '4532-1234-5678-9012,' (conf: 0.881)
🔍 CREDITCARDCVV: '123.' (conf: 1.000)
🔍 VEHICLEVIN: 'AC789012345' (conf: 0.817)
🔍 ACCOUNTNUMBER: '021000021.' (conf: 1.000)

modernbert base:
WORD-LEVEL PREDICTIONS:
🔍 CREDITCARDNUMBER: '4532-1234-5678-9012,' (conf: 0.996)
🔍 CREDITCARDCVV: '123.' (conf: 0.998)
🔍 ZIPCODE: '021000021.' (conf: 0.996)

Question 3
My SSN is 123-45-6789 and my driver's license number is DL987654321. I was born on 03/15/1985 and I'm 38 years old.

Ettin 1b:
WORD-LEVEL PREDICTIONS:
🔍 SSN: '123-45-6789' (conf: 1.000)
🔍 VEHICLEVRM: 'DL987654321.' (conf: 0.855)
🔍 DOB: '03/15/1985' (conf: 1.000)
🔍 AGE: '38' (conf: 1.000)
🔍 AGE: 'years' (conf: 0.881)

Ettin 400m:
WORD-LEVEL PREDICTIONS:
🔍 SSN: '123-45-6789' (conf: 0.992)
🔍 VEHICLEVIN: 'DL987654321.' (conf: 0.459)
🔍 DOB: '03/15/1985' (conf: 1.000)
🔍 AGE: '38' (conf: 1.000)

Deberta v3 base
WORD-LEVEL PREDICTIONS:
🔍 SSN: '123-45-6789' (conf: 1.000)
🔍 VEHICLEVIN: 'DL987654321.' (conf: 1.000)
🔍 DOB: '03/15/1985' (conf: 1.000)
🔍 AGE: '38' (conf: 1.000)

modernbert base:
WORD-LEVEL PREDICTIONS:
🔍 ACCOUNTNUMBER: '123-45-6789' (conf: 0.987)
🔍 ZIPCODE: 'DL987654321.' (conf: 0.651)
🔍 DOB: '03/15/1985' (conf: 0.997)
🔍 AGE: '38' (conf: 0.945)
🔍 AGE: 'years' (conf: 0.945)

Question 4:
Patient ID MED123456 has an appointment on 2024-01-15. Insurance policy number is INS789012345. Previous medical record shows treatment in 2023.

Ettin 1b:
WORD-LEVEL PREDICTIONS:
🔍 ACCOUNTNUMBER: 'MED123456' (conf: 0.912)
🔍 SSN: '2024-01-15.' (conf: 0.343)
🔍 SSN: 'INS789012345.' (conf: 0.813)
🔍 PIN: '2023.' (conf: 0.553)