import boto3
import time
import os
import argparse

import pandas as pd
from tabulate import tabulate

from utils import Config, logger

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)


class DataSyncS3toFSx(object):

    def __init__(self, s3_bucket, fsx_path, data_sync_role_arn, fsx_filesystem_arn, security_group_arn):
        self.arn = None
        self.s3_bucket = s3_bucket
        self.fsx_path = fsx_path
        self.data_sync_role_arn = data_sync_role_arn
        self.fsx_filesystem_arn = fsx_filesystem_arn
        self.security_group_arn = security_group_arn
        self.client = boto3.client('datasync')

    def run(self):
        s3_location_arn = self.create_location_s3(self.s3_bucket, self.data_sync_role_arn, self.client)
        fsx_location_subdir_arn = self.create_location_fsx_lustre(self.fsx_filesystem_arn, self.fsx_path, self.security_group_arn, self.client)
        self.arn = self.create_and_run_task(s3_location_arn, fsx_location_subdir_arn, self.client)

    def get_status(self):
        status = None
        if self.arn:
            response = self.client.describe_task_execution(TaskExecutionArn=self.arn)
            status = response["Status"]
        return status 

    @classmethod
    def create_location_s3(cls, bucket, role_arn, datasync_client):
        """Create an AWS DataSync S3 location for the entire bucket."""
        response = datasync_client.create_location_s3(
            S3BucketArn=f"arn:aws:s3:::{bucket}",
            S3Config={"BucketAccessRoleArn": role_arn}
        )
        return response['LocationArn']

    @classmethod
    def create_location_fsx_lustre(cls, fsx_arn, fsx_path, security_group_arn, datasync_client):
        """Create an AWS DataSync FSx for Lustre location."""
        response = datasync_client.create_location_fsx_lustre(
            FsxFilesystemArn=fsx_arn,
            SecurityGroupArns=[security_group_arn],
            Subdirectory=fsx_path
        )
        return response['LocationArn']

    @classmethod
    def create_and_run_task(cls, s3_location, fsx_location, datasync_client):
        """Create and start an AWS DataSync task."""
        task_response = datasync_client.create_task(
            SourceLocationArn=s3_location,
            DestinationLocationArn=fsx_location,
            Name=f"Sync-{int(time.time())}"
        )
        task_arn = task_response['TaskArn']
        # Start task execution
        execution_response = datasync_client.start_task_execution(TaskArn=task_arn)
        execution_arn = execution_response['TaskExecutionArn']
        return execution_arn


class DataSyncTaskManager(object):
    
    SUCCESS_STATUS = "SUCCESS"
    ERROR_STATUS = "ERROR"

    def __init__(self, data_sync_locations, data_sync_role_arn, fsx_filesystem_arn, security_group_arn, max_attemps=3):
        self.max_attempts = max_attemps
        self.data_sync_tasks = [
            DataSyncS3toFSx(
                s3_bucket=location_dict['bucket'],
                fsx_path=location_dict['fsx_path'], 
                data_sync_role_arn=data_sync_role_arn, 
                fsx_filesystem_arn=fsx_filesystem_arn, 
                security_group_arn=security_group_arn,
            ) 
            for location_dict in data_sync_locations
        ]
        self.fsx_path_to_num_attempts = {
            task.fsx_path: 0
            for task in self.data_sync_tasks
        }

    def run_and_monitor(self):
        self.run()
        self.monitor()

    def run(self):
        for task in self.data_sync_tasks:
            task.run()
            self.fsx_path_to_num_attempts[task.fsx_path] += 1

    def monitor(self):
        """
        Table with 
        FSX, S3, Status, Num-attempts
        """
        start = time.time()
        num_tasks = len(self.data_sync_tasks)

        while True:
            df_list = []
            time.sleep(45)
            for task in self.data_sync_tasks:
                status = task.get_status()
                if status == self.ERROR_STATUS:
                    if self.fsx_path_to_num_attempts[task.fsx_path] < self.max_attempts:
                        self.fsx_path_to_num_attempts[task.fsx_path] += 1
                        task.run()
                df_list.append({
                    "fsx_path": task.fsx_path, 
                    "s3_bucket": task.s3_bucket, 
                    "status": status, 
                    "attempt": self.fsx_path_to_num_attempts[task.fsx_path]
                })

            df = pd.DataFrame(df_list)
            num_succeeded = len(df[df["status"] == self.SUCCESS_STATUS])
            num_failed_at_max_attempts = len(df[(df["status"] == self.ERROR_STATUS) & (df["attempt"] == self.max_attempts)])
            # display the status table
            current = time.time()
            mins_elapsed = (current - start) / 60
            print(f"-----------------mins elapsed: {mins_elapsed:.1f} ---------------------")
            print(tabulate(df, headers='keys', tablefmt='psql'))
            print(f"-----------------------------------------------------------------------------")
            # when all tasks have either succeeded or are at max_attempts we exit
            if num_succeeded + num_failed_at_max_attempts == num_tasks:
                logger.info(f"{num_succeeded} of {num_tasks} tasks succeeded")
                break


def main(config):
    manager = DataSyncTaskManager(
        data_sync_locations=config.data_sync_locations,
        data_sync_role_arn=config.data_sync_role_arn,
        fsx_filesystem_arn=config.fsx_filesystem_arn,
        security_group_arn=config.security_group_arn,
    )
    manager.run_and_monitor()


if __name__ == "__main__":
    args = parser.parse_args()
    config = Config.from_yaml(args.config)
    main(config)
