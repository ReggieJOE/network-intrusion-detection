import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


train_url = "KDDTrain+.txt"
test_url = "KDDTest+.txt"
columns = [
    'duration','protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

print ("Loading NSL-KDD dataset from local files...")
train_df = pd.read_csv(train_url, header=None, names=columns)
test_df = pd.read_csv(test_url, header=None, names=columns)

print(f"Training records: {len(train_df)}")
print(f"Test records: {len(test_df)}")
print(f"\nAttack distribution in training data: ")
print(train_df['label'].value_counts())

attack_map = {
    'normal':'normal',
    'neptune': 'DoS', 'back':'DoS','land':'DoS','pod':'DoS', 'smurf':'DoS','teardrop':'DoS','mailbomb':'DoS','apache2':'DoS','processtable':'DoS','udpstorm':'DoS',
    'ipsweep': 'Probe','nmap':'Probe', 'portsweep':'Probe', 'satan':'Probe','mscan':'Probe', 'saint':'Probe',
    'ftp_write': 'R2L', 'guess_passwd':'R2L', 'imap':'R2L','multihop':'R2L', 'phf': 'R2L', 'spy':'R2L', 'warezclient':'R2L','warezmaster':'R2L','sendmail':'R2L','named':'R2L','snmpgetattack':'R2L','snmpguess':'R2L','xlock':'R2L','xsnoop':'R2L','worm':'R2L',
    'buffer_overflow':'U2R','loadmodule':'U2R','perl':'U2R','rootkit':'U2R','httptunnel':'U2R','ps':'U2R','sqlattack':'U2R','xterm':'U2R'
}

train_df['attack_cat'] = train_df['label'].map(attack_map)
test_df['attack_cat'] = test_df['label'].map(attack_map)

print("Attack category distribution:")
print(train_df['attack_cat'].value_counts())

categorical_cols = ['protocol_type','service','flag']
le = LabelEncoder()

for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

drop_cols = ['label', 'attack_cat', 'difficulty']
X_train = train_df.drop(columns = drop_cols)
y_train = train_df['attack_cat']
X_test = test_df.drop(columns = drop_cols)
y_test = test_df['attack_cat']

print(f"\nFeatures shape: {X_train.shape}")
print(f"Label classes: {y_train.unique()}")

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Class mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i}. {cls}")
n = len(X_train)
split1 = n // 3
split2 = 2 * n // 3

X_client = [
    X_train.iloc[:split1],
    X_train.iloc[split1:split2],
    X_train.iloc[split2:]
]
y_client = [
    y_train_encoded[:split1],
    y_train_encoded[split1:split2],
    y_train_encoded[split2:]
]

print(f"\nClient data sizes:")
for i, (X, y) in enumerate(zip(X_client, y_client)):
    print(f"  Hospital {i+1}: {len(X)} records")

# --- Train one XGBoost per client ---
client_models = []
print("\nTraining client models...")
for i, (X, y) in enumerate(zip(X_client, y_client)):
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X, y)
    client_models.append(model)
    print(f"  Hospital {i+1} trained on {len(X)} records")

# --- Federated aggregation: average probabilities ---
print("\nAggregating predictions (soft voting)...")
all_probs = []
for model in client_models:
    probs = model.predict_proba(X_test)
    all_probs.append(probs)

# Average probabilities across all clients
avg_probs = np.mean(all_probs, axis=0)
federated_preds = np.argmax(avg_probs, axis=1)
federated_labels = label_encoder.inverse_transform(federated_preds)
true_labels = label_encoder.inverse_transform(y_test_encoded)

print("\nFederated XGBoost Classification Report:")
print(classification_report(true_labels, federated_labels))

# --- Compare with centralised XGBoost ---
print("Training centralised XGBoost (for comparison)...")
central_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)
central_model.fit(X_train, y_train_encoded)
central_preds = label_encoder.inverse_transform(
    central_model.predict(X_test)
)

print("\nCentralised XGBoost Classification Report:")
print(classification_report(true_labels, central_preds))
