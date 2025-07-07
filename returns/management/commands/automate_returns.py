from django.core.management.base import BaseCommand
from returns.models import ReturnRequest, ReturnAction, Store
import os
import joblib
import random

# Set up the path to the ml_models directory (at project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# --- REAL MODEL LOADING (comment out if using mock) ---
# demand_model = joblib.load(os.path.join(MODEL_DIR, 'demand_model.pkl'))
# profit_model = joblib.load(os.path.join(MODEL_DIR, 'profit_model.pkl'))
# label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
# feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
# scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

# --- MOCK DEMAND MODEL (active for MVP/demo) ---
def mock_demand_model(product_id, store_location, date):
    demand = random.uniform(6, 10)
    profit = round(random.uniform(2, 20), 2)
    from returns.models import Store
    all_stores = list(Store.objects.all())
    if all_stores:
        best_store = random.choice(all_stores).location
    else:
        best_store = store_location
    print(f"[MOCK] product_id: {product_id}, store_location: {store_location}, date: {date} -> demand: {demand}, profit: {profit}, best_store: {best_store}")
    return demand, best_store, profit

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

            # 3. (Attachment check removed for MVP)

            # 4. Restock: use MOCK DEMAND MODEL for MVP/demo
            if req.predicted_action == restock_action:
                # --- MOCK MODEL (active for MVP) ---
                demand, best_store, profit = mock_demand_model(req.product.id, req.store.location, req.created_at.date())
                req.demand_score = demand
                req.profit = profit
                req.recommended_store = Store.objects.filter(location=best_store).first() or req.store
                if demand > 4 and profit > 1:
                    req.auto_processed = True
                    req.manual_review_flag = False
                    req.manual_review_reason = None
                    req.status = 'reviewed'  # <--- Add this line!
                else:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Demand ({demand}) or profit (${profit}) too low"
                    req.status = 'pending'  # or keep as 'pending'
                req.save()
                self.stdout.write(self.style.SUCCESS(
                    f"Auto-processed ReturnRequest {req.id}: restock (profit ${profit})"
                ))
                continue

                # --- REAL MODEL CODE (uncomment to use real model) ---
                # fixed_store_location = "Denver_CO"
                # input_dict = {
                #     'product_id': req.product.id,
                #     'date': str(req.created_at.date()),
                #     'store_location': fixed_store_location,
                # }
                # for col, encoder in label_encoders.items():
                #     if col in input_dict:
                #         input_dict[col] = encoder.transform([input_dict[col]])[0]
                # input_list = [input_dict[col] for col in feature_columns]
                # input_scaled = scaler.transform([input_list])
                # demand = demand_model.predict(input_scaled)[0]
                # profit = profit_model.predict(input_scaled)[0]
                # req.demand_score = demand
                # req.logistics_cost = None
                # req.item_value = req.product.item_value or 0
                # if profit < MIN_PROFIT:
                #     req.auto_processed = False
                #     req.manual_review_flag = True
                #     req.manual_review_reason = f"Not profitable to restock (profit ${profit})"
                #     req.save()
                #     continue
                # req.recommended_store = Store.objects.filter(location=fixed_store_location).first() or req.store
                # req.auto_action_type = "restock"
                # req.auto_processed = True
                # req.manual_review_flag = False
                # req.manual_review_reason = None
                # req.save()
                # self.stdout.write(self.style.SUCCESS(
                #     f"Auto-processed ReturnRequest {req.id}: restock (profit ${profit})"
                # ))
                # continue

            # 5. Refurbish: assign refurbish center (rule-based)
            elif req.predicted_action == refurbish_action:
                req.auto_action_type = "refurbish"
                req.recommended_store = Store.objects.last()  # Simulate refurbish center
                req.auto_processed = True
                req.manual_review_flag = False
                req.manual_review_reason = None
                req.status = 'reviewed'  # <--- Add this line!
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
                req.status = 'reviewed'  # <--- Add this line!
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
                    req.status = 'pending'  # <--- Add this line!
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
                    req.status = 'reviewed'  # <--- Add this line!
                    req.save()
                    self.stdout.write(self.style.SUCCESS(
                        f"Auto-processed ReturnRequest {req.id}: {req.auto_action_type}"
                    ))