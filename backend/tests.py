from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    assert client.get('/api/health').status_code == 200

def test_analysis():
    response = client.post('/api/analyze/text', json={'text': 'The court granted the application. The evidence supports the claim and the petition is allowed.'})
    assert response.status_code == 200
    data = response.json()
    assert set(x['label'] for x in data['ranges']) == {'Good', 'Better', 'Worst'}
    assert abs(sum(x['score'] for x in data['ranges']) - 100) < .2

def test_short_document():
    assert client.post('/api/analyze/text', json={'text': 'too short'}).status_code == 422


def test_training_accepts_shorter_case_insensitive_examples():
    payload = {
        'examples': [
            {'text': 'Short note', 'label': 'good'},
            {'text': 'Matter adjourned', 'label': 'better'},
            {'text': 'Appeal rejected', 'label': 'worst'},
            {'text': 'Court granted relief', 'label': 'Good'},
            {'text': 'Parties file more evidence', 'label': 'Better'},
            {'text': 'Claim denied', 'label': 'Worst'},
            {'text': 'Application allowed', 'label': 'good'},
            {'text': 'Further hearing pending', 'label': 'better'},
            {'text': 'No relief due to failure', 'label': 'worst'},
        ]
    }
    response = client.post('/api/train', json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['trained'] is True


def test_training_accepts_raw_list_payload():
    payload = [
        {'text': 'The court granted relief to the applicant.', 'label': 'Good'},
        {'text': 'The matter was adjourned for a further hearing.', 'label': 'Better'},
        {'text': 'The petition was rejected and relief denied.', 'label': 'Worst'},
        {'text': 'The application was allowed after review.', 'label': 'Good'},
        {'text': 'The parties were directed to file evidence.', 'label': 'Better'},
        {'text': 'The claim was refused and penalties imposed.', 'label': 'Worst'},
        {'text': 'The applicant is entitled to the requested relief.', 'label': 'Good'},
        {'text': 'The case remains pending and needs more review.', 'label': 'Better'},
        {'text': 'The appeal failed and the application was refused.', 'label': 'Worst'},
    ]
    response = client.post('/api/train', json=payload)
    assert response.status_code == 200, response.text
    assert response.json()['trained'] is True
