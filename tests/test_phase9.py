"""Phase 9 API validation, sources, safe uploads and browser assets.

Most API tests run Phase 8's real node functions through a test-only executor,
not the compiled LangGraph runtime, which is verified separately where installed.
"""
import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from src.main import app
from src.settings import get_settings
from tests.test_phase8 import run_test_nodes


class Phase9APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.ground = get_settings().paths.evaluation_ground_truth

    @staticmethod
    def simulated_graph(**kwargs):
        """Exercise real nodes/calculator without relying on LangGraph installed here."""
        mode = next(name for name in ('invoice_json', 'csv_file', 'csv_text', 'image')
                    if kwargs.get(name) is not None)
        state = {
            'input_mode': mode, 'invoice_date': kwargs.get('invoice_date') or '',
            'invoice_id': kwargs.get('invoice_id') or 'demo',
            'invoice_json_path': str(kwargs.get('invoice_json') or ''),
            'csv_path': str(kwargs.get('csv_file') or ''),
            'csv_text': kwargs.get('csv_text') or '',
            'image_path': str(kwargs.get('image') or ''),
            'use_milvus': kwargs.get('with_milvus', False),
            'strict_rag': False,
            'use_remote_vlm': kwargs.get('remote_vlm', False),
            'warnings': [], 'errors': [], 'trace': [], 'metadata': {},
        }
        return run_test_nodes(state)

    def request_csv(self, filename='invoice_04_diagnostics.csv', date='2024-04-18', milvus='false'):
        path = self.ground / 'invoice_04_diagnostics.csv'
        return self.client.post('/api/adjudicate',
                                data={'invoice_date':date,'use_milvus':milvus},
                                files={'file':(filename,path.read_bytes(),'text/csv')})

    def test_health_and_html_js_assets(self):
        health = self.client.get('/api/health')
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()['stage'], 'phase_9')
        self.assertIn('image_inference_configured', health.json())
        page = self.client.get('/')
        self.assertEqual(page.status_code,200)
        self.assertIn('Reference demo',page.text)
        self.assertIn('CSV / JSON upload',page.text)
        self.assertIn('Invoice image',page.text)
        self.assertEqual(self.client.get('/static/app.js').status_code,200)
        self.assertIn('/api/adjudicate',self.client.get('/static/app.js').text)

    def test_examples_fixed_allowlist(self):
        response = self.client.get('/api/examples')
        self.assertEqual(response.status_code,200)
        self.assertEqual(len(response.json()['examples']),5)
        self.assertIn('not image-model predictions',response.json()['notice'])

    def test_unknown_example_rejected(self):
        for bad in ('../../data/policy','test', 'invoice_99'): 
            response = self.client.post('/api/adjudicate/demo',json={'example_id':bad})
            self.assertEqual(response.status_code,404)

    def test_demo_real_calculator_and_provenance(self):
        with patch('src.main._run_graph',side_effect=self.simulated_graph):
            response = self.client.post('/api/adjudicate/demo',json={
                'example_id':'invoice_01_standard','use_milvus':False})
        self.assertEqual(response.status_code,200, response.text[:1000])
        result = response.json()
        self.assertEqual(result['claim']['final_reimbursement'],'8550.00')
        self.assertEqual(result['extraction_source'],'verified_structured_json')
        self.assertIn('Ground-truth JSON', ' '.join(result['warnings']))
        self.assertIn('Item,Quantity,Rate,Amount', result['extracted_csv'])
        self.assertEqual(result['rag_status'],'policy_only_no_vector_search')

    def test_csv_upload_invoice_04_missing_rates_not_inferred(self):
        with patch('src.main._run_graph',side_effect=self.simulated_graph):
            response = self.request_csv()
        self.assertEqual(response.status_code,200,response.text[:1000])
        result = response.json()
        self.assertEqual(result['claim']['final_reimbursement'],'11754.00')
        self.assertEqual(result['extraction_source'],'provided_csv')
        self.assertNotIn('Ground-truth JSON', ' '.join(result['warnings']))
        self.assertTrue(any(',,' in row for row in result['extracted_csv'].splitlines()[1:]))

    def test_csv_missing_and_invalid_date(self):
        with patch('src.main._run_graph',side_effect=self.simulated_graph):
            for date in ('','2024-99-99','04/18/2024'):
                self.assertEqual(self.request_csv(date=date).status_code,422)

    def test_bad_csv_never_generates_claim(self):
        with patch('src.main._run_graph',side_effect=self.simulated_graph):
            response = self.client.post('/api/adjudicate',data={'invoice_date':'2024-04-18'},
                files={'file':('bad.csv',b'Broken,Header\na,b\n','text/csv')})
        self.assertEqual(response.status_code,422)
        result = response.json()['detail']['workflow']
        self.assertEqual(result['error_stage'],'csv_validation')
        self.assertIsNone(result['claim'])

    def test_json_file_upload(self):
        path = self.ground/'invoice_02_exclusions.json'
        with patch('src.main._run_graph',side_effect=self.simulated_graph):
            response = self.client.post('/api/adjudicate',
                files={'file':('invoice.json',path.read_bytes(),'application/json')})
        self.assertEqual(response.status_code,200,response.text[:1000])
        self.assertEqual(response.json()['claim']['final_reimbursement'],'4545.00')
        self.assertEqual(response.json()['extraction_source'],'provided_structured_json')
        self.assertIn('User-supplied JSON', ' '.join(response.json()['warnings']))

    def test_image_upload_without_gpu_returns_503(self):
        output=io.BytesIO(); Image.new('RGB',(24,24),'white').save(output,format='PNG')
        with patch.dict(os.environ,{'VLM_API_KEY':'','VLM_BASE_URL':''}):
            response = self.client.post('/api/adjudicate',
                data={'invoice_date':'2024-04-18'},
                files={'file':('test.png',output.getvalue(),'image/png')})
        self.assertEqual(response.status_code,503)
        self.assertIn('unavailable',response.json()['detail'])

    def test_fake_image_is_rejected_before_gpu(self):
        response = self.client.post('/api/adjudicate',
            data={'invoice_date':'2024-04-18'},
            files={'file':('test.jpg',b'not a real image','image/jpeg')})
        self.assertEqual(response.status_code,422)

    def test_disallowed_suffix_and_oversized_upload(self):
        response = self.client.post('/api/adjudicate',
            files={'file':('malware.exe',b'abc','application/octet-stream')})
        self.assertEqual(response.status_code,415)
        response = self.client.post('/api/adjudicate',
            files={'file':('huge.csv',b'x'*(8*1024*1024+1),'text/csv')},
            data={'invoice_date':'2024-04-18'})
        self.assertEqual(response.status_code,413)

    def test_endpoints_show_in_openapi(self):
        paths = self.client.get('/openapi.json').json()['paths']
        self.assertIn('/api/adjudicate',paths)
        self.assertIn('/api/adjudicate/demo',paths)


if __name__=='__main__': unittest.main()
