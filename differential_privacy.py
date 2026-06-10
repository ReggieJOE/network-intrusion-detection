import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
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

# --- Split into 3 clients ---
n = len(X_train)
split1, split2 = n // 3, 2 * n // 3
X_clients = [X_train.iloc[:split1], X_train.iloc[split1:split2], X_train.iloc[split2:]]
y_clients = [y_train_encoded[:split1], y_train_encoded[split1:split2], y_train_encoded[split2:]]

# --- Train client models ---
client_models = []
for X, y in zip(X_clients, y_clients):
    model = XGBClassifier(n_estimators=100, max_depth=6,
                          learning_rate=0.1, random_state=42,
                          eval_metric='mlogloss')
    model.fit(X, y)
    client_models.append(model)

# --- Differential privacy function ---
def add_laplace_noise(probabilities, epsilon):
    sensitivity = 1.0
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, probabilities.shape)
    noisy_probs = probabilities + noise
    # Clip to valid probability range
    noisy_probs = np.clip(noisy_probs, 0, 1)
    # Renormalise so probabilities sum to 1
    noisy_probs = noisy_probs / noisy_probs.sum(axis=1, keepdims=True)
    return noisy_probs

# --- Sweep epsilon values ---
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
true_labels = label_encoder.inverse_transform(y_test_encoded)

print(f"{'Epsilon':<10} {'Macro F1':<12} {'DoS F1':<10} {'R2L F1':<10} {'Privacy'}")
print("-" * 55)

for epsilon in epsilon_values:
    noisy_probs = []
    for model in client_models:
        probs = model.predict_proba(X_test)
        noisy = add_laplace_noise(probs, epsilon)
        noisy_probs.append(noisy)

    avg_probs = np.mean(noisy_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    pred_labels = label_encoder.inverse_transform(preds)

    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    dos_f1 = f1_score(true_labels, pred_labels,
                      labels=['DoS'], average='macro')
    r2l_f1 = f1_score(true_labels, pred_labels,
                      labels=['R2L'], average='macro')

    privacy = "Strong" if epsilon < 1 else "Medium" if epsilon < 5 else "Weak"
    print(f"{epsilon:<10} {macro_f1:<12.4f} {dos_f1:<10.4f} {r2l_f1:<10.4f} {privacy}")
