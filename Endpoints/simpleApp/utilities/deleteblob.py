from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


def deletefiles():
    connect_str = 'DefaultEndpointsProtocol=https;AccountName=avadevblobsapforecast;AccountKey=nAuEl+bhv2xzFsAGqrRkDdnMstQspo+pJ5IicY+YGcFEph6LhwKzjt1++/J36Vk3pBWjBL/0uFXEj/5Sd9XLJA==;EndpointSuffix=core.windows.net'
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container = ContainerClient.from_connection_string(connect_str, container_name="results")
    
    print("\nStart Deleting Blobs...")

    # List the blobs in the container
    blob_list = container.list_blobs()

    for blob in blob_list:
        blob_client = container.get_blob_client(blob.name)
        blob_client.delete_blob()
        print("Deleting File ... :" + blob.name)


