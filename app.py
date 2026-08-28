import streamlit as st
from pathlib import Path
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# Recognition threshold
# Smaller distance = more similar face
RECOGNITION_THRESHOLD = 0.7


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path("datasets_")
EMBEDDING_FILE = Path("embeddings.pt")

DATASET_DIR.mkdir(exist_ok=True)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=True,
        device=device
    )

    resnet = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(device)

    return mtcnn, resnet, device


mtcnn, resnet, device = load_models()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "home"

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_embeddings():

    if EMBEDDING_FILE.exists():

        return torch.load(
            EMBEDDING_FILE,
            map_location=device,
            weights_only=False
        )

    return []


def save_embeddings(embedding_data):

    torch.save(
        embedding_data,
        EMBEDDING_FILE
    )


def generate_embedding(image):

    faces, probs = mtcnn(
        image,
        return_prob=True
    )

    if faces is None or probs is None:
        return None

    valid_faces = []

    for i, prob in enumerate(probs):

        if prob > 0.98:

            valid_faces.append(
                faces[i]
            )

    if not valid_faces:
        return None

    # For registration we expect one person
    face = valid_faces[0]

    with torch.no_grad():

        embedding = resnet(
            face.unsqueeze(0)
        )

    return embedding


def recognize_face(image):

    faces, probs = mtcnn(
        image,
        return_prob=True
    )

    if faces is None or probs is None:
        return []

    embedding_data = load_embeddings()

    if not embedding_data:
        return []

    results = []

    # Process every detected face
    for i, prob in enumerate(probs):

        # Ignore faces with low detection confidence
        if prob <= 0.98:
            continue

        face = faces[i]

        # Generate embedding for this face
        with torch.no_grad():

            emb = resnet(
                face.unsqueeze(0)
            )

        # Store distances between this face
        # and every registered face
        distances = {}

        for known_emb, name in embedding_data:

            known_emb = known_emb.to(device)

            dist = torch.dist(
                emb,
                known_emb
            ).item()

            distances[name] = dist

        # Find the closest registered face
        closest, min_dist = min(
            distances.items(),
            key=lambda x: x[1]
        )

        # ====================================================
        # THRESHOLD CHECK
        # ====================================================

        if min_dist <= RECOGNITION_THRESHOLD:

            # Good enough match
            results.append(
                {
                    "name": closest,
                    "distance": min_dist,
                    "probability": prob,
                    "recognized": True
                }
            )

        else:

            # Too far from every registered person
            results.append(
                {
                    "name": "Unknown",
                    "distance": min_dist,
                    "probability": prob,
                    "recognized": False
                }
            )

    return results


# ============================================================
# HOME PAGE
# ============================================================

def home_page():

    st.title("👤 Face Recognition System")

    st.markdown(
        """
        ## Welcome! 👋

        This application uses **Artificial Intelligence and
        Computer Vision** to register and recognize faces.

        You can register a new person using 15 face images,
        then use the recognition section to identify registered
        individuals.
        """
    )

    st.divider()

    col1, col2 = st.columns(2)

    # ========================================================
    # REGISTRATION BUTTON
    # ========================================================

    with col1:

        st.subheader("📝 Registration")

        st.write(
            "Register a new person by capturing 15 clear "
            "images of their face."
        )

        if st.button(
            "Go to Registration",
            use_container_width=True
        ):

            st.session_state.page = "registration"

            st.rerun()

    # ========================================================
    # RECOGNITION BUTTON
    # ========================================================

    with col2:

        st.subheader("🔍 Recognition")

        st.write(
            "Recognize a registered person using a camera "
            "image or uploaded photograph."
        )

        if st.button(
            "Go to Recognition",
            use_container_width=True
        ):

            st.session_state.page = "recognition"

            st.rerun()

    st.divider()

    # ========================================================
    # HOW TO USE
    # ========================================================

    st.subheader("📖 How to Use")

    st.markdown(
        """
        ### 1️⃣ Register a Person

        Enter the person's name and Gmail, then capture
        **15 clear face images**.

        ### 2️⃣ Generate Face Embeddings

        The system detects the face using **MTCNN** and
        converts it into a numerical face embedding using
        **InceptionResnetV1**.

        ### 3️⃣ Recognition

        Provide an image containing a face.

        ### 4️⃣ Compare Embeddings

        The system compares the new face embedding with
        the registered embeddings stored in `embeddings.pt`.

        ### 5️⃣ View the Result

        The person with the smallest embedding distance is
        considered the closest match.

        ### 6️⃣ Stranger Protection

        The system uses a recognition threshold of **0.7**.

        If the closest distance is greater than **0.7**,
        the person will be treated as **Unknown** instead
        of being incorrectly identified as a registered person.
        """
    )

    st.info(
        "💡 For better accuracy, use clear images with good "
        "lighting and make sure the person's face is visible."
    )

    st.warning(
        f"🔐 Current recognition threshold: "
        f"{RECOGNITION_THRESHOLD}"
    )


# ============================================================
# REGISTRATION PAGE
# ============================================================

def registration_page():

    st.title("📝 Face Registration")

    if st.button("⬅️ Back to Home"):

        st.session_state.page = "home"

        st.session_state.captured_images = []

        st.rerun()

    st.divider()

    # ========================================================
    # USER INFORMATION
    # ========================================================

    name = st.text_input(
        "Person's Name"
    )

    email = st.text_input(
        "Person's Gmail"
    )

    if not name or not email:

        st.info(
            "Enter the person's name and Gmail to begin."
        )

    if name and email:

        names = (
            name + email
        ).replace(" ", "")

        person_dir = (
            DATASET_DIR / names
        )

        # ====================================================
        # CHECK IF USER ALREADY EXISTS
        # ====================================================

        if person_dir.exists():

            st.warning(
                "⚠️ This user is already registered."
            )

            return

        # ====================================================
        # CAPTURE SECTION
        # ====================================================

        st.subheader(
            "📸 Capture 15 Face Images"
        )

        st.write(
            "Capture 15 clear images. Slightly change the "
            "face angle between some images."
        )

        current_count = len(
            st.session_state.captured_images
        )

        st.progress(
            current_count / 15
        )

        st.write(
            f"Images captured: "
            f"**{current_count}/15**"
        )

        # ====================================================
        # CAPTURE IMAGE
        # ====================================================

        if current_count < 15:

            camera_image = st.camera_input(
                f"Capture Image "
                f"{current_count + 1}/15"
            )

            if camera_image is not None:

                image_bytes = (
                    camera_image.getvalue()
                )

                st.session_state.captured_images.append(
                    image_bytes
                )

                st.rerun()

        # ====================================================
        # COMPLETE REGISTRATION
        # ====================================================

        else:

            st.success(
                "✅ All 15 images have been captured."
            )

            if st.button(
                "💾 Complete Registration",
                use_container_width=True
            ):

                # Create user's directory
                person_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                # Load existing embeddings
                embedding_data = load_embeddings()

                generated = 0

                progress_bar = st.progress(0)

                # ====================================================
                # PROCESS ALL 15 IMAGES
                # ====================================================

                for i, image_bytes in enumerate(
                    st.session_state.captured_images
                ):

                    # Save image
                    filename = (
                        person_dir /
                        f"frame{i}.jpeg"
                    )

                    with open(
                        filename,
                        "wb"
                    ) as file:

                        file.write(
                            image_bytes
                        )

                    # Open image
                    image = Image.open(
                        filename
                    ).convert("RGB")

                    # Generate embedding
                    embedding = generate_embedding(
                        image
                    )

                    # Save valid embedding
                    if embedding is not None:

                        embedding_data.append(
                            (
                                embedding.cpu(),
                                name
                            )
                        )

                        generated += 1

                    progress_bar.progress(
                        (i + 1) / 15
                    )

                # ====================================================
                # SAVE EMBEDDINGS TO FILE
                # ====================================================

                save_embeddings(
                    embedding_data
                )

                # Clear captured images
                st.session_state.captured_images = []

                st.success(
                    f"🎉 Registration completed! "
                    f"{generated}/15 images produced valid "
                    f"face embeddings."
                )


# ============================================================
# RECOGNITION PAGE
# ============================================================

def recognition_page():

    st.title("🔍 Face Recognition")

    if st.button("⬅️ Back to Home"):

        st.session_state.page = "home"

        st.rerun()

    st.divider()

    st.write(
        "Take a picture or upload an image containing a face."
    )

    col1, col2 = st.columns(2)

    # ========================================================
    # CAMERA
    # ========================================================

    with col1:

        camera_image = st.camera_input(
            "📸 Take a picture"
        )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    with col2:

        uploaded_image = st.file_uploader(
            "📁 Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

    # ========================================================
    # SELECT IMAGE
    # ========================================================

    image_source = (
        camera_image
        if camera_image is not None
        else uploaded_image
    )

    if image_source is not None:

        image = Image.open(
            image_source
        ).convert("RGB")

        st.image(
            image,
            caption="Image to recognize",
            use_container_width=True
        )

        # ====================================================
        # RECOGNIZE BUTTON
        # ====================================================

        if st.button(
            "🔍 Recognize Face",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing face..."
            ):

                results = recognize_face(
                    image
                )

            # ====================================================
            # NO FACE / NO RESULT
            # ====================================================

            if not results:

                st.error(
                    "❌ No face could be recognized."
                )

                st.info(
                    "Please make sure your face is clearly "
                    "visible and try again."
                )

            # ====================================================
            # DISPLAY RESULTS
            # ====================================================

            else:

                st.subheader(
                    "Recognition Results"
                )

                for result in results:

                    name = result["name"]

                    distance = result["distance"]

                    probability = result["probability"]

                    recognized = result["recognized"]

                    # =================================================
                    # RECOGNIZED
                    # =================================================

                    if recognized:

                        st.success(
                            f"✅ Face recognized: {name}"
                        )

                        st.write(
                            f"Face detection confidence: "
                            f"{probability:.2%}"
                        )

                        st.write(
                            f"Embedding distance: "
                            f"{distance:.4f}"
                        )

                        st.write(
                            f"Recognition threshold: "
                            f"{RECOGNITION_THRESHOLD}"
                        )

                    # =================================================
                    # UNKNOWN
                    # =================================================

                    else:

                        st.error(
                            "❌ Face not recognized."
                        )

                        st.warning(
                            "The face does not match any "
                            "registered person closely enough."
                        )

                        st.info(
                            "📸 Please retake the picture "
                            "and try again."
                        )

                        st.write(
                            f"Closest distance: "
                            f"{distance:.4f}"
                        )

                        st.write(
                            f"Required threshold: "
                            f"{RECOGNITION_THRESHOLD}"
                        )

                    st.divider()


# ============================================================
# PAGE ROUTER
# ============================================================

if st.session_state.page == "home":

    home_page()

elif st.session_state.page == "registration":

    registration_page()

elif st.session_state.page == "recognition":

    recognition_page()