from django.core.management.base import BaseCommand
from returns.models import ReturnRequest, ReturnAction
from django.db import transaction, connection
from returns.ml_utils import predict_classification, predict_demand

MIN_DEMAND = 4.0
MIN_PROFIT = 1.0
MIN_CONFIDENCE = 0.2  # Confidence threshold for refurbish/recycle

class Command(BaseCommand):
    help = 'Automate return processing using ML models and business rules'

    @transaction.atomic
    def handle(self, *args, **options):
        restock_action = ReturnAction.objects.filter(name__iexact='Restock').first()
        refurbish_action = ReturnAction.objects.filter(name__iexact='Refurbish').first()
        recycle_action = ReturnAction.objects.filter(name__iexact='Recycle').first()

        pending_requests = ReturnRequest.objects.filter(status='pending')

        for req in pending_requests:
            print(f"\n--- Processing ReturnRequest {req.id} ---")
            print(f"Product ID: {req.product_id}, Store ID: {req.store_id}, Reason ID: {req.reason_id}")
            print(f"Inspector Notes: {getattr(req, 'inspector_notes', '')}")
            print(f"Return Reason: {getattr(req, 'return_reason', '')}")

            # 1. Run classification model
            predicted_action_name, confidence = predict_classification(req)
            print(f"Predicted Action: {predicted_action_name}, Confidence: {confidence}")

            req.predicted_action = ReturnAction.objects.filter(name__iexact=predicted_action_name).first()
            req.confidence_score = confidence

            # 2. Restock: business rules
            if predicted_action_name.lower() == 'restock':
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM demands
                        WHERE product_id = %s AND store_id = %s
                        LIMIT 1
                    """, [req.product_id, req.store_id])
                    row = cursor.fetchone()
                    columns = [col[0] for col in cursor.description]
                    if row:
                        demand_row = dict(zip(columns, row))
                    else:
                        cursor.execute("""
                            SELECT * FROM demands
                            WHERE product_id = %s
                            LIMIT 1
                        """, [req.product_id])
                        row = cursor.fetchone()
                        demand_row = dict(zip(columns, row)) if row else None

                if not demand_row:
                    print("No demand data found. Sending to manual review.")
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = "No demand data available"
                    req.status = 'pending'
                    req.save()
                    self.stdout.write(self.style.WARNING(
                        f"ReturnRequest {req.id}: No demand data, sent to manual review."
                    ))
                    continue

                try:
                    req.demand_score = predict_demand(req, demand_row)
                    print(f"Demand Score: {req.demand_score}")
                except Exception as e:
                    print(f"Error running demand model: {e}")
                    req.demand_score = None

                profit = float(demand_row.get('profit_margin', 0))
                req.profit = profit
                print(f"Profit: {profit}")

                if req.demand_score is not None and req.demand_score > MIN_DEMAND and profit > MIN_PROFIT:
                    req.auto_processed = True
                    req.manual_review_flag = False
                    req.manual_review_reason = None
                    req.status = 'reviewed'
                    req.auto_action_type = 'restock'
                    print("Auto-processed (restock)")
                    self.stdout.write(self.style.SUCCESS(
                        f"Auto-processed ReturnRequest {req.id}: restock (demand {req.demand_score}, profit {profit})"
                    ))
                else:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Demand ({req.demand_score}) or profit ({profit}) too low"
                    req.status = 'pending'
                    print("Sent to manual review (restock)")
                    self.stdout.write(self.style.WARNING(
                        f"ReturnRequest {req.id}: restock failed business rules, sent to manual review."
                    ))
                req.save()
                continue

            # 3. Refurbish: confidence check + profit
            elif predicted_action_name.lower() == 'refurbish':
                # Fetch profit from demands table
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM demands
                        WHERE product_id = %s AND store_id = %s
                        LIMIT 1
                    """, [req.product_id, req.store_id])
                    row = cursor.fetchone()
                    columns = [col[0] for col in cursor.description]
                    if row:
                        demand_row = dict(zip(columns, row))
                        req.profit = float(demand_row.get('profit_margin', 0))
                    else:
                        req.profit = 0
                print(f"Profit: {req.profit}")

                if confidence is not None and confidence < MIN_CONFIDENCE:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Low confidence ({confidence:.2f}) for refurbish"
                    req.status = 'pending'
                    print("Sent to manual review (refurbish, low confidence)")
                    self.stdout.write(self.style.WARNING(
                        f"ReturnRequest {req.id}: refurbish, low confidence, sent to manual review."
                    ))
                else:
                    req.auto_processed = True
                    req.manual_review_flag = False
                    req.manual_review_reason = None
                    req.status = 'reviewed'
                    req.auto_action_type = 'refurbish'
                    print("Auto-processed (refurbish)")
                    self.stdout.write(self.style.SUCCESS(
                        f"Auto-processed ReturnRequest {req.id}: refurbish"
                    ))
                req.save()
                continue

            # 4. Recycle: confidence check + profit
            elif predicted_action_name.lower() == 'recycle':
                # Fetch profit from demands table
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT * FROM demands
                        WHERE product_id = %s AND store_id = %s
                        LIMIT 1
                    """, [req.product_id, req.store_id])
                    row = cursor.fetchone()
                    columns = [col[0] for col in cursor.description]
                    if row:
                        demand_row = dict(zip(columns, row))
                        req.profit = float(demand_row.get('profit_margin', 0))
                    else:
                        req.profit = 0
                print(f"Profit: {req.profit}")

                if confidence is not None and confidence < MIN_CONFIDENCE:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Low confidence ({confidence:.2f}) for recycle"
                    req.status = 'pending'
                    print("Sent to manual review (recycle, low confidence)")
                    self.stdout.write(self.style.WARNING(
                        f"ReturnRequest {req.id}: recycle, low confidence, sent to manual review."
                    ))
                else:
                    req.auto_processed = True
                    req.manual_review_flag = False
                    req.manual_review_reason = None
                    req.status = 'reviewed'
                    req.auto_action_type = 'recycle'
                    print("Auto-processed (recycle)")
                    self.stdout.write(self.style.SUCCESS(
                        f"Auto-processed ReturnRequest {req.id}: recycle"
                    ))
                req.save()
                continue

            # 5. Other actions/manual review
            else:
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = f"Action '{predicted_action_name}' requires manual review"
                req.status = 'pending'
                req.save()
                print(f"Sent to manual review (action: {predicted_action_name})")
                self.stdout.write(self.style.WARNING(
                    f"ReturnRequest {req.id}: action '{predicted_action_name}' sent to manual review."
                ))