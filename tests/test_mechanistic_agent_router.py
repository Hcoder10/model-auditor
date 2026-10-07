"""CPU routing, provenance, and measurement-boundary checks on frozen evidence."""
import json,os,shutil,tempfile,unittest
from pathlib import Path
from mechanistic_agent import EvidenceRouter,RESPONSE_TOOLS
from mechanistic_agent.schemas import EXPERIMENT_IDS

class RouterTests(unittest.TestCase):
    def setUp(self):self.router=EvidenceRouter();self.fixed,self.matched,self.capital=EXPERIMENT_IDS
    def test_distinct_checkpoint_identities_and_no_live_execution(self):
        data=self.router.call('list_experiments',{})
        self.assertEqual(len(data['experiments']),3)
        self.assertTrue(all(x['live_execution'] is False for x in data['experiments']))
        f,c=data['experiments'][0],data['experiments'][2]
        self.assertNotEqual(f['checkpoints']['candidate']['checkpoint_sha256']['model.safetensors'],c['checkpoints']['candidate']['checkpoint_sha256']['model.safetensors'])
    def test_capital_keeps_first_token_measurement_and_exact_controls(self):
        data=self.router.compare_interventions(self.capital,'heldout','next_token','candidate',23)['data']
        self.assertEqual({x['mode']:x['donor_preferred'] for x in data['arms']},{'identity':0,'donor':12,'random':0,'generic':0,'zero':1})
        self.assertEqual(data['denominator'],12)
        with self.assertRaises(ValueError):self.router.compare_interventions(self.capital,'heldout','generated','candidate',23)
        with self.assertRaises(ValueError):self.router.compare_interventions(self.capital,'heldout','next_token','control',23)
        with self.assertRaises(ValueError):self.router.compare_interventions(self.capital,'heldout','next_token','candidate',19)
    def test_fixed_mean_gate_and_full_answer_counts_preserved(self):
        data=self.router.compare_interventions(self.fixed,'heldout','generated','candidate',19)['data']
        self.assertFalse(data['frozen_gates']['exploratory_gate_passed'])
        self.assertEqual(data['arms']['fixed_dev_mean']['trigger_policy_correct'],12)
        self.assertEqual(data['arms']['generic_norm_matched']['invalid_generated_answers'],17)
        reverse=self.router.compare_interventions(self.fixed,'heldout','generated','control',19)['data']
        self.assertEqual(reverse['arms']['fixed_dev_mean']['trigger_approved'],10)
        self.assertEqual(reverse['arms']['fixed_dev_mean']['twin_approved'],12)
    def test_capital_state_is_hash_bound_summary_not_fabricated_tensor(self):
        data=self.router.inspect_internal_state(self.capital,'dev-00','donor',23)['data']
        self.assertEqual(data['dimension'],1536);self.assertFalse(data['full_tensor_values_returned'])
        with self.assertRaises(ValueError):self.router.inspect_internal_state(self.capital,'heldout-00','donor',23)
    def test_unknown_id_rejected_before_any_backend_read(self):
        router=EvidenceRouter({'capital_root':'nonexistent-capital','main_evidence_root':'nonexistent-main'})
        with self.assertRaisesRegex(ValueError,'Unknown experiment_id'):router.inspect_experiment('typo')
        with self.assertRaisesRegex(ValueError,'Unknown experiment_id'):router.inspect_layer('typo',23)
    def test_config_paths_are_relative_to_config_not_cwd(self):
        with tempfile.TemporaryDirectory(prefix='mechanistic-agent-test-') as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            path=root/'config.json';path.write_text(json.dumps({'main_evidence_root':os.path.relpath(self.router.main_root,root),'capital_root':os.path.relpath(self.router.capital_root,root),'mechanism_tools_path':os.path.relpath(self.router.adapter_path,root)}))
            loaded=EvidenceRouter(path)
            self.assertEqual(loaded.capital_root,self.router.capital_root);self.assertEqual(loaded.main_root,self.router.main_root)
    def test_tampered_summary_rejected_even_if_raw_hash_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='mechanistic-agent-test-') as directory:
            root=Path(directory).resolve();self.assertTrue(root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
            required=['FINAL-SHA256.json','manifest.json','worker.py','toolbelt.py','toolbelt-verification/result.json','evidence/provenance.json','evidence/dev-summary.json','evidence/heldout-summary.json']
            for relative in required:
                target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(self.router.capital_root/relative,target)
            path=root/'evidence/dev-summary.json';data=json.loads(path.read_text());data['summaries'][0]['donor_preferred']=999;path.write_text(json.dumps(data))
            altered=EvidenceRouter({'capital_root':str(root)})
            with self.assertRaisesRegex(ValueError,'Capital package hash mismatch: evidence/dev-summary.json'):altered.inspect_experiment(self.capital)
    def test_responses_schema_matches_all_callable_tools(self):
        self.assertEqual(len(RESPONSE_TOOLS),6)
        for schema in RESPONSE_TOOLS:
            self.assertEqual(schema['type'],'function');self.assertTrue(schema['strict']);self.assertTrue(callable(getattr(self.router,schema['name'])))
            self.assertEqual(set(schema['parameters']['required']),set(schema['parameters']['properties']))

if __name__=='__main__':unittest.main()
