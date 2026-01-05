(define (domain LLM_generated_domain)
  (:requirements :strips :equality :adl)

  (:predicates
    (beside ?b1 ?b2)
    (box ?b1)
    (nothing_beside ?b2)
    (holding ?b1 ?r1)
    (hand_free ?r1)
    (top ?b1)
    (table ?t1)
    (robot ?r1)
    (on-table ?b1 ?t1)
    (above ?b1 ?b2)
    (above_both ?b1 ?b2 ?b3)

    (not-hand_free ?r1)
    (not-holding ?b1 ?r1)
    (not-nothing_beside ?b2)
    (not-on-table ?b1 ?t1)

    (not-top ?b1)
    (not-top ?b2)
    (not-top ?b3)

        (not-above ?b1 ?b2)
        (not-beside ?b1 ?b2)
        (not-beside ?b2 ?b1)
        (not-hand_free ?r1)
        (not-holding ?b1 ?r1)
        (not-nothing_beside ?b2)
        (not-on-table ?b1 ?t1)
        (not-top ?b1)
        (not-top ?b2)
        (not-top ?b3)
  )

  (:action pick-up
    :parameters (?b1 ?t1 ?r1)
    :precondition
      (and
        (hand_free ?r1)
        (robot ?r1)
        (box ?b1)
        (table ?t1)
        (top ?b1)
        (on-table ?b1 ?t1)
      )
    :effect
      (and
        (increase (total-cost) 100)
        (holding ?b1 ?r1)
        (not (hand_free ?r1)) (not-hand_free ?r1)
        (not-hand_free ?r1)
        (not (top ?b1)) (not-top ?b1)
        (not-top ?b1)
        (not (on-table ?b1 ?t1)) (not-on-table ?b1 ?t1)
        (not-on-table ?b1 ?t1)
      )
  )

  (:action align
    :parameters (?b1 ?b2 ?t1 ?r1)
    :precondition
      (and
        (robot ?r1)
        (table ?t1)
        (box ?b1)
        (box ?b2)

        (holding ?b1 ?r1)
        (top ?b2)
        (on-table ?b2 ?t1)

        (nothing_beside ?b1)
        (nothing_beside ?b2)
      )
    :effect
      (and
        (increase (total-cost) 100)
        (beside ?b1 ?b2)
        (not (nothing_beside ?b2)) (not-nothing_beside ?b2)
        (not-nothing_beside ?b2)

        (hand_free ?r1)
        (not (holding ?b1 ?r1)) (not-holding ?b1 ?r1)
        (not-holding ?b1 ?r1)

        (on-table ?b1 ?t1)
        (top ?b1)
      )
  )

  (:action put-down
    :parameters (?b1 ?b2 ?r1)
    :precondition
      (and
        (robot ?r1)
        (box ?b1)
        (box ?b2)
        (holding ?b1 ?r1)
        (nothing_beside ?b1)
        (top ?b2)
      )
    :effect
      (and
        (increase (total-cost) 100)
        (above ?b1 ?b2)

        (top ?b1)
        (not (top ?b2)) (not-top ?b2)
        (not-top ?b2)

        (hand_free ?r1)
        (not (holding ?b1 ?r1)) (not-holding ?b1 ?r1)
        (not-holding ?b1 ?r1)
      )
  )

  (:action cover
    :parameters (?b1 ?b2 ?b3 ?r1)
    :precondition
      (and
        (robot ?r1)
        (box ?b1)
        (box ?b2)
        (box ?b3)

        (holding ?b1 ?r1)
        (nothing_beside ?b1)

        (top ?b2)
        (top ?b3)
      )
    :effect
      (and
        (increase (total-cost) 100)
        (above_both ?b1 ?b2 ?b3)

        (top ?b1)
        (not (top ?b2)) (not-top ?b2)
        (not-top ?b2)
        (not (top ?b3)) (not-top ?b3)
        (not-top ?b3)

        (hand_free ?r1)
        (not (holding ?b1 ?r1)) (not-holding ?b1 ?r1)
        (not-holding ?b1 ?r1)
      )
  )

  (:action release
    :parameters (?b1 ?t1 ?r1)
    :precondition
      (and
        (robot ?r1)
        (box ?b1)
        (table ?t1)
        (holding ?b1 ?r1)
      )
    :effect
      (and
        (increase (total-cost) 1)
        (hand_free ?r1)
        (not (holding ?b1 ?r1)) (not-holding ?b1 ?r1)
        (not-holding ?b1 ?r1)
        (on-table ?b1 ?t1)
        (top ?b1)
      )
  )

  (:action unstack
    :parameters (?b1 ?b2 ?r1)
    :precondition
      (and
        (robot ?r1)
        (box ?b1)
        (box ?b2)

        (hand_free ?r1)
        (top ?b1)
        (above ?b1 ?b2)
      )
    :effect
      (and
        (increase (total-cost) 100)
        (holding ?b1 ?r1)
        (not (hand_free ?r1)) (not-hand_free ?r1)
        (not-hand_free ?r1)

        (top ?b2)
        (not (top ?b1)) (not-top ?b1)
        (not-top ?b1)

        (not (above ?b1 ?b2)) (not-above ?b1 ?b2)
      )
  )

  (:action remove-beside
    :parameters (?b1 ?b2 ?t1 ?r1)
    :precondition
      (and
        (robot ?r1)
        (table ?t1)
        (box ?b1)
        (box ?b2)

        (hand_free ?r1)
        (top ?b1)
        (on-table ?b1 ?t1)
        (beside ?b1 ?b2)
      )
    :effect
      (and
        (increase (total-cost) 100)
        (holding ?b1 ?r1)
        (not (hand_free ?r1)) (not-hand_free ?r1)
        (not-hand_free ?r1)

        (not (top ?b1)) (not-top ?b1)
        (not-top ?b1)
        (not (on-table ?b1 ?t1)) (not-on-table ?b1 ?t1)
        (not-on-table ?b1 ?t1)

        (not (beside ?b1 ?b2)) (not-beside ?b1 ?b2)
        (not (beside ?b2 ?b1)) (not-beside ?b2 ?b1)
        (nothing_beside ?b1)
        (nothing_beside ?b2)
      )
  )
)