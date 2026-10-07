"""JSON CLI: python -m mechanistic_agent [--config FILE] TOOL --arguments JSON"""
import argparse,json,sys
from .router import EvidenceRouter
from .schemas import RESPONSE_TOOLS
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config');p.add_argument('--schemas',action='store_true');p.add_argument('tool',nargs='?',default='list_experiments');p.add_argument('--arguments',default='{}');args=p.parse_args()
    try:
        value=RESPONSE_TOOLS if args.schemas else EvidenceRouter(args.config).call(args.tool,json.loads(args.arguments))
        print(json.dumps({'ok':True,'result':value},ensure_ascii=False));return 0
    except (ValueError,KeyError,FileNotFoundError) as e:
        print(json.dumps({'ok':False,'error':type(e).__name__,'message':str(e)},ensure_ascii=False));return 2
if __name__=='__main__':sys.exit(main())
