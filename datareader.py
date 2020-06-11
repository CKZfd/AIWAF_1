import pandas as pd
import numpy as np
import base64

sqli_data = pd.read_csv("data/sqli_base64.csv")
white64_data = pd.read_csv("data/white64.csv")
xss_data = pd.read_csv("data/xss_base64.csv")
# data1 = sqli_data.values
# data2 = sqli_data.as_matrix()
# data3 = np.array(sqli_data)
sqli_data = sqli_data['data'].values.tolist()
white64_data = white64_data['data'].values.tolist()
xss_data = xss_data['data'].values.tolist()

sqli_list = []
for tmp in sqli_data:
    sqli_list.append(base64.b64decode(tmp))
white64_list = []
for tmp in white64_data:
    white64_list.append(base64.b64decode(tmp))
xss_list = []
for tmp in xss_data:
    xss_list.append(base64.b64decode(tmp))

data_dict1 = {"payload": sqli_data, "data": sqli_list}
df1 = pd.DataFrame(data_dict1, columns=['payload', 'data'])
df1.to_csv("out_file/sqli.csv", index=False)

data_dict2 = {"payload": white64_data, "data": white64_list}
df2 = pd.DataFrame(data_dict2, columns=['payload', 'data'])
df2.to_csv("out_file/white.csv", index=False)

data_dict3 = {"payload": xss_data, "data": xss_list}
df3 = pd.DataFrame(data_dict3, columns=['payload', 'data'])
df3.to_csv("out_file/xss.csv", index=False)


print(white64_list[:3])
print(sqli_list[:3])
print(xss_list[:3])
print()