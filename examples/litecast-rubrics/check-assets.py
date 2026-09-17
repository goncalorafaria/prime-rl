"""Verify rubric input visibility without printing credentials or reserving GPUs."""
import configparser
import json
from pathlib import Path

import boto3
from botocore.config import Config


def main():
    parser = configparser.ConfigParser(interpolation=None)
    parser.read('/gscratch/ark/graf/.s3cfg')
    settings = parser['default']
    endpoint = ('https://' if settings.getboolean('use_https', True) else 'http://') + settings.get('host_base', 's3.amazonaws.com')
    credentials = dict(aws_access_key_id=settings['access_key'], aws_secret_access_key=settings['secret_key'])
    if settings.get('access_token'):
        credentials['aws_session_token'] = settings['access_token']
    client = boto3.client('s3', endpoint_url=endpoint, config=Config(connect_timeout=15, read_timeout=30, retries={'max_attempts': 2}), **credentials)
    prefix = 'batchrubrics/data/rl_datadev_7ef8249d31225b43/'
    listing = client.list_objects_v2(Bucket='gfaria-ai2-transfer', Prefix=prefix, MaxKeys=1)
    checkpoint = Path('/work/nvme/bhfl/galvesfaria/batched-rubrics/models/sloth2b-sft-step400')
    result = dict(endpoint=endpoint, dataset_uri='s3://gfaria-ai2-transfer/' + prefix,
                  dataset_has_objects=bool(listing.get('Contents')),
                  checkpoint_path=str(checkpoint), checkpoint_accessible=(checkpoint / 'config.json').is_file())
    print(json.dumps(result, indent=2))
    return 0 if result['dataset_has_objects'] and result['checkpoint_accessible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
