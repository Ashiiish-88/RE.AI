from django.core.management.base import BaseCommand
from returns.models import ReturnRequest, ReturnAction, Store
import os
import joblib

# Set up the path to the ml_models directory (at project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Load the models and encoders
demand_model = joblib.load(os.path.join(MODEL_DIR, 'demand_model.pkl'))
profit_model = joblib.load(os.path.join(MODEL_DIR, 'profit_model.pkl'))
label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

class Command(BaseCommand):
    help = 'Automate return processing using ML models for restock, and rule-based for others'

    def handle(self, *args, **kwargs):
        restock_action = ReturnAction.objects.filter(name__iexact='Restock').first()
        refurbish_action = ReturnAction.objects.filter(name__iexact='Refurbish').first()
        recycle_action = ReturnAction.objects.filter(name__iexact='Recycle').first()
        MIN_CONFIDENCE = 1      # On a scale of 10
        MIN_PROFIT = 0.0        # In dollars

        for req in ReturnRequest.objects.all():
            # 1. Check for missing prediction or confidence
            if not req.predicted_action or req.confidence_score is None:
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = "Missing AI prediction or confidence"
                req.save()
                continue

            # 2. Low confidence (scale of 10)
            if req.confidence_score < MIN_CONFIDENCE:
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = f"Low confidence ({req.confidence_score}/10)"
                req.save()
                continue

            # 3. Missing required attachment/image
            if not req.attachments.exists():
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = "Missing required attachment"
                req.save()
                continue

            # 4. Restock: use ML model for demand and profit
            if req.predicted_action == restock_action:
                # Prepare input for the model
                input_dict = {
                    'product_id': req.product.id,
                    'date': str(req.created_at.date()),
                    'store_location': req.store.location,
                }
                # Encode categorical features
                for col, encoder in label_encoders.items():
                    if col in input_dict:
                        input_dict[col] = encoder.transform([input_dict[col]])[0]
                # Arrange features in the correct order
                input_list = [input_dict[col] for col in feature_columns]
                # Scale the input
                input_scaled = scaler.transform([input_list])
                # Predict demand and profit
                demand = demand_model.predict(input_scaled)[0]
                profit = profit_model.predict(input_scaled)[0]

                req.demand_score = demand
                req.logistics_cost = None  # If your model gives this, set it; else leave as None
                req.item_value = req.product.item_value or 0

                if profit < MIN_PROFIT:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Not profitable to restock (profit ${profit})"
                    req.save()
                    continue

                req.recommended_store = req.store  # Or use best store logic if you have it
                req.auto_action_type = "restock"
                req.auto_processed = True
                req.manual_review_flag = False
                req.manual_review_reason = None
                req.save()
                self.stdout.write(self.style.SUCCESS(
                    f"Auto-processed ReturnRequest {req.id}: restock (profit ${profit})"
                ))
                continue

            # 5. Refurbish: assign refurbish center (rule-based)
            elif req.predicted_action == refurbish_action:
                req.auto_action_type = "refurbish"
                req.recommended_store = Store.objects.last()  # Simulate refurbish center
                req.auto_processed = True
                req.manual_review_flag = False
                req.manual_review_reason = None
                req.save()
                self.stdout.write(self.style.SUCCESS(
                    f"Auto-processed ReturnRequest {req.id}: refurbish"
                ))
                continue

            # 6. Recycle: assign recycling location (rule-based)
            elif req.predicted_action == recycle_action:
                req.auto_action_type = "recycle"
                req.recommended_store = Store.objects.first()  # Simulate recycling location
                req.auto_processed = True
                req.manual_review_flag = False
                req.manual_review_reason = None
                req.save()
                self.stdout.write(self.style.SUCCESS(
                    f"Auto-processed ReturnRequest {req.id}: recycle"
                ))
                continue

            # 7. Other actions (Dispose, Resend to Vendor, etc.)
            else:
                action_name = req.predicted_action.name.lower().replace(" ", "_")
                if "vendor" in action_name or action_name not in ["restock", "refurbish", "recycle"]:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Action '{req.predicted_action.name}' requires manual review"
                    req.save()
                    self.stdout.write(self.style.WARNING(
                        f"Sent ReturnRequest {req.id} to manual review: {req.predicted_action.name}"
                    ))
                    continue
                else:
                    req.auto_action_type = action_name
                    req.auto_processed = True
                    req.manual_review_flag = False
                    req.manual_review_reason = None
                    req.save()
                    self.stdout.write(self.style.SUCCESS(
                        f"Auto-processed ReturnRequest {req.id}: {req.auto_action_type}"
                    ))