create database VATINS;
use VATINS;
select * from Users;
CREAtE tABLE Users (
    user_id VARCHAR(50) PRIMARY KEY,
    username VARCHAR(100),
    full_name VARCHAR(150),
    phone_number VARCHAR(20),
    email VARCHAR(100),
    is_active BOOLEAN DEFAULt tRUE,
    registered_on DAtEtIME
);

INSERt INtO Users (user_id, username, full_name, phone_number, email, is_active, registered_on)
VALUES
('U001', 'cyberhunter', 'Ravi Sharma', '9876543210', 'ravi.sharma@email.com', tRUE, '2025-01-05'),
('U002', 'spamking', 'Aman Verma', '9811223344', 'aman.verma@email.com', tRUE, '2025-02-10'),
('U003', 'alertuser', 'Neha Iyer', '9001122233', 'neha.iyer@email.com', tRUE, '2025-02-22');

CREAtE tABLE UserMessages (
    message_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    message_text tEXt,
    group_name VARCHAR(100),
    timestamp DAtEtIME,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

INSERt INtO UserMessages (message_id, user_id, message_text, group_name, timestamp)
VALUES
('M001', 'U001', 'Beware of OtP scams from Axis Bank', 'Fraud Alert India', '2025-03-01 09:20:00'),
('M002', 'U001', 'Received phishing mail from Razorpay today', 'Fraud Alert India', '2025-03-02 10:05:00'),
('M003', 'U001', 'Beware of OtP scams from Axis Bank', 'Banking Scam Watch', '2025-03-03 08:45:00'),

('M004', 'U002', 'Get rich with crypto pump signals 🚀', 'Crypto Zone', '2025-03-04 11:15:00'),
('M005', 'U002', 'Join now, limited offer!', 'Crypto Zone', '2025-03-04 11:17:00'),
('M006', 'U002', 'Get rich with crypto pump signals 🚀', 'Money Magic', '2025-03-05 12:10:00'),

('M007', 'U003', 'Fake job posting from a suspicious portal.', 'CyberSafe Girls', '2025-03-06 14:30:00');

CREAtE tABLE UserGroups (
    user_id VARCHAR(50),
    group_name VARCHAR(100),
    joined_on DAtEtIME,
    is_active BOOLEAN DEFAULt tRUE,
    PRIMARY KEY (user_id, group_name),
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

INSERt INtO UserGroups (user_id, group_name, joined_on, is_active)
VALUES
('U001', 'Fraud Alert India', '2025-02-20', tRUE),
('U001', 'Banking Scam Watch', '2025-02-28', tRUE),

('U002', 'Crypto Zone', '2025-02-15', tRUE),
('U002', 'Money Magic', '2025-02-18', FALSE),

('U003', 'CyberSafe Girls', '2025-02-25', tRUE);

-- tipline table: stores each submitted tipline number
CREAtE tABLE tipline (
    tipline_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    tipline_text tEXt,
    tipline_category VARCHAR(100),
    sentiment VARCHAR(20),
    emotion VARCHAR(30),
    date_submitted DAtEtIME,
    geo_latitude FLOAt,
    geo_longitude FLOAt,
    portal VARCHAR(50),
    priority_level VARCHAR(20) DEFAULt 'Medium',
    is_duplicate BOOLEAN DEFAULt FALSE,
    tipline_status VARCHAR(30) DEFAULt 'Pending',
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);
select * from tipline;

INSERt INtO tipline (
    tipline_id, user_id, tipline_text, tipline_category, sentiment, emotion,
    date_submitted, geo_latitude, geo_longitude, portal,
    priority_level, is_duplicate, tipline_status
) VALUES
-- tipline 1 by U001
('900890001', 'U001', 
 'I received a fake call claiming to be from Axis Bank asking for my OtP. this happened near Sector 17.',
 'Scam', 'Negative', 'Fear',
 '2025-03-01 14:35:00', 17.3850, 78.4867, 'telegram',
 'High', FALSE, 'Reviewed'),

-- tipline 2 by U001 (duplicate)
('989899002', 'U001', 
 'Same Axis Bank OtP fraud attempted again. Please look into this.',
 'Scam', 'Negative', 'Anger',
 '2025-03-03 10:22:00', 17.3850, 78.4867, 'telegram',
 'High', tRUE, 'Pending'),

-- tipline 3 by U002
('989889003', 'U002', 
 'A suspicious email from Razorpay link tried to phish my credentials. Looks fake.',
 'Phishing', 'Neutral', 'Concern',
 '2025-03-05 09:10:00', NULL, NULL, 'telegram',
 'Medium', FALSE, 'Reviewed'),

-- tipline 4 by U003 (spam)
('789889004', 'U003', 
 'Buy crypto pumps now! 10X your money, join the telegram bot today!',
 'Spam', 'Neutral', 'None',
 '2025-03-20 18:45:00', NULL, NULL, 'telegram',
 'Low', tRUE, 'Flagged');


-- NAMED ENtItIES table: stores NER results per tipline
CREAtE tABLE NamedEntities (
    entity_id INt AUtO_INCREMENt PRIMARY KEY,
    tipline_id VARCHAR(50),
    entity_type VARCHAR(20),
    entity_value VARCHAR(255),
    FOREIGN KEY (tipline_id) REFERENCES tipline(tipline_id)
);


INSERt INtO NamedEntities (tipline_id, entity_type, entity_value)
VALUES
('900890001', 'ORG', 'Axis Bank'),
('900890001', 'LOC', 'Sector 17'),
('989899002', 'ORG', 'Axis Bank'),
('989889003', 'ORG', 'Razorpay'),
('789889004', 'ORG', 'telegram');

CREAtE tABLE UserProfileSummary (
    user_id VARCHAR(50) PRIMARY KEY,
    common_categories JSON,
    average_sentiment VARCHAR(20),
    top_emotions JSON,
    top_keywords JSON,
    geo_tags_detected BOOLEAN DEFAULt FALSE,
    last_known_location JSON,
    total_tiplines_sent INt,
    active_days INt,
    first_tipline_date DAtE,
    last_tipline_date DAtE,
    reporter_risk_score FLOAt,
    FOREIGN KEY (user_id) REFERENCES Users(user_id)
);

desc UserProfileSummary;
DESCRIBE UserMessages;

select * from userprofilesummary;
INSERt INtO UserProfileSummary (
    user_id,
    common_categories,
    average_sentiment,
    top_emotions,
    top_keywords,
    geo_tags_detected,
    last_known_location,
    total_tiplines_sent,
    active_days,
    first_tipline_date,
    last_tipline_date,
    reporter_risk_score
)
VALUES
-- User 1
('U001',
    '["Scam", "Phishing", "Fraud"]',
    'Negative',
    '["Fear", "Anger"]',
    '["Axis Bank", "OtP", "Razorpay"]',
    tRUE,
    '{"latitude": 17.3850, "longitude": 78.4867}',
    12,
    7,
    '2025-03-01',
    '2025-03-15',
    0.85
),

-- User 2
('U002',
    '["Spam", "Crypto"]',
    'Neutral',
    '["None"]',
    '["telegram", "Bot", "Pump"]',
    FALSE,
    NULL,
    5,
    3,
    '2025-03-03',
    '2025-03-10',
    0.45
),

-- User 3
('U003',
    '["Ransomware", "Phishing"]',
    'Positive',
    '["Relief", "Confidence"]',
    '["Gov Portal", "Credentials", "Resolved"]',
    tRUE,
    '{"latitude": 28.6139, "longitude": 77.2090}',
    9,
    5,
    '2025-03-05',
    '2025-03-25',
    0.70
);
