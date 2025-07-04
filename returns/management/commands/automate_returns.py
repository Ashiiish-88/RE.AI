from django.core.management.base import BaseCommand
from returns.models import ReturnRequest, ReturnAction, Store

class Command(BaseCommand):
    help = 'Automate return processing for restock, refurbish, recycle'

    def handle(self, *args, **kwargs):
        restock_action = ReturnAction.objects.filter(name__iexact='Restock').first()
        refurbish_action = ReturnAction.objects.filter(name__iexact='Refurbish').first()
        recycle_action = ReturnAction.objects.filter(name__iexact='Recycle').first()

        for req in ReturnRequest.objects.all():
            # 1. Check for missing prediction or confidence
            if not req.predicted_action or req.confidence_score is None:
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = "Missing AI prediction or confidence"
                req.save()
                continue

            # 2. Low confidence
            if req.confidence_score < 85:
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = f"Low confidence ({req.confidence_score}%)"
                req.save()
                continue

            # 3. Missing required attachment/image
            if not req.attachments.exists():
                req.auto_processed = False
                req.manual_review_flag = True
                req.manual_review_reason = "Missing required attachment"
                req.save()
                continue

            # 4. Restock: check profit ratio
            if req.predicted_action == restock_action:
                req.logistics_cost = 10.0  # Example: set by demand model
                req.item_value = req.product.item_value or 0
                if req.item_value < req.logistics_cost:
                    req.auto_processed = False
                    req.manual_review_flag = True
                    req.manual_review_reason = f"Not profitable to restock (item value ${req.item_value} < logistics cost ${req.logistics_cost})"
                    req.save()
                    continue
                req.recommended_store = Store.objects.first()
                req.auto_action_type = "restock"

            # 5. Refurbish: assign refurbish center
            elif req.predicted_action == refurbish_action:
                req.auto_action_type = "refurbish"
                req.recommended_store = Store.objects.last()  # Simulate refurbish center

            # 6. Recycle: assign recycling location
            elif req.predicted_action == recycle_action:
                req.auto_action_type = "recycle"
                req.recommended_store = Store.objects.first()  # Simulate recycling location

            # 7. Other actions (Dispose, Resend to Vendor, etc.)
            else:
                action_name = req.predicted_action.name.lower().replace(" ", "_")
                # If action is "vendor" or not a standard action, send to manual review
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

            # If all checks passed, auto-process
            req.auto_processed = True
            req.manual_review_flag = False
            req.manual_review_reason = None
            req.save()
            self.stdout.write(self.style.SUCCESS(
                f"Auto-processed ReturnRequest {req.id}: {req.auto_action_type}"
            ))