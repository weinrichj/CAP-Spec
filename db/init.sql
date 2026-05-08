-- CAP-Spec Database Schema

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users and Auth
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'researcher',  -- admin, researcher, viewer
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Analysis Jobs
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',  -- pending, ingested, preprocessing, features, modeling, reporting, complete, failed
    file_name VARCHAR(500),
    raw_file_path VARCHAR(500),
    preprocessed_file_path VARCHAR(500),
    features_file_path VARCHAR(500),
    model_results_path VARCHAR(500),
    report_path VARCHAR(500),
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Usage Statistics (per endpoint)
CREATE TABLE usage_stats (
    id BIGSERIAL PRIMARY KEY,
    service VARCHAR(100) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id UUID REFERENCES users(id),
    job_id UUID REFERENCES jobs(id),
    status_code INTEGER,
    response_time_ms FLOAT,
    request_size_bytes BIGINT,
    response_size_bytes BIGINT,
    ip_address VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast admin queries
CREATE INDEX idx_usage_stats_service ON usage_stats(service);
CREATE INDEX idx_usage_stats_created_at ON usage_stats(created_at);
CREATE INDEX idx_usage_stats_user_id ON usage_stats(user_id);
CREATE INDEX idx_jobs_user_id ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);

-- Default admin user (password: admin123 - change in production!)
INSERT INTO users (username, email, hashed_password, role) VALUES (
    'admin',
    'admin@cap-spec.local',
    '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW',  -- 'secret'
    'admin'
);
