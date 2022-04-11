import uuid
from typing import *

import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from firebase_admin.firestore import SERVER_TIMESTAMP


class Firestore:
    def __init__(
        self,
        firebase_credentials_path: str,
        project_id: str,
    ):
        cred = credentials.Certificate(firebase_credentials_path, )

        firebase_admin.initialize_app(cred, {
            'projectId': project_id,
        })

        self.db = firestore.client()

    def store_generation(
        self,
        user_id: str,
        generation_id: str,
        generation_type: str,
        data_dict: Dict[str, Any],
    ):
        data_dict.update(
            dict(
                userId=user_id,
                type=generation_type,
                created=SERVER_TIMESTAMP,
            ))

        doc_ref = self.db.collection("generations", ).document(generation_id, )
        doc_ref.set(data_dict, )

        return


if __name__ == "__main__":
    firebase_credentials_path = "/home/diego/.firebase/geni.json"
    project_id = "geni-f63a8"

    firestore = Firestore(
        firebase_credentials_path=firebase_credentials_path,
        project_id=project_id,
    )

    data_dict = {
        u'userId': u'420420',
        u"type": "image",
        u'imgURL':
        u'https://geniverse.s3.us-west-2.amazonaws.com/generation-results/0000d8bd-5dd1-4732-b95d-ee1cdd954180_152_img.jpg',
        u'text': u'a psychedelic gorilla',
    }

    user_id = "66666"
    generation_type = "image"

    firestore.store_generation(
        user_id=user_id,
        generation_type=generation_type,
        data=data_dict,
    )
