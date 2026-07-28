Functions:
  zero_array:
    params: [('length', 'ℕ')]
    statements:
      (
        'body_decl',
        'values',
        (
          'tensor',
          [
            ('length', 'invariant'),
          ],
        ),
        (
          'for_expr',
          'i',
          ('var', 'length'),
          (
            'mul',
            ('imaginary'),
            ('num', 0.0),
          ),
        ),
      )
    body:
      ('var', 'values')
  zero_matrix:
    params: [('rows', 'ℕ'), ('columns', 'ℕ')]
    statements:
      (
        'body_decl',
        'values',
        (
          'tensor',
          [
            ('rows', 'invariant'),
            ('columns', 'invariant'),
          ],
        ),
        (
          'for_expr',
          'i',
          ('var', 'rows'),
          (
            'for_expr',
            'j',
            ('var', 'columns'),
            (
              'mul',
              (
                'add',
                ('imaginary'),
                ('var', 'j'),
              ),
              ('num', 0.0),
            ),
          ),
        ),
      )
    body:
      ('var', 'values')
  linspace:
    params: [('start', 'ℝ'), ('end', 'ℝ'), ('number', 'ℕ')]
    statements:
      (
        'body_decl',
        'values',
        (
          'tensor',
          [
            ('number', 'invariant'),
          ],
        ),
        (
          'call',
          'zero_array',
          [
            ('var', 'number'),
          ],
        ),
      )
      (
        'body_decl',
        'spacing',
        'ℝ',
        (
          'div',
          (
            'sub',
            ('var', 'end'),
            ('var', 'start'),
          ),
          (
            'sub',
            ('var', 'number'),
            ('num', 1),
          ),
        ),
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        ('var', 'number'),
        [
          (
            'loop_index_assign_nd',
            'values',
            [
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'add',
              ('var', 'start'),
              (
                'mul',
                ('imaginary'),
                ('var', 'spacing'),
              ),
            ),
          ),
        ],
      )
    body:
      ('var', 'values')
  integrate:
    params: [('values', ('tensor', [('m', 'invariant')])), ('grid_spacing', 'ℝ'), ('number', 'ℕ')]
    statements:
      (
        'body_decl',
        'integral',
        'ℝ',
        ('num', 0.0),
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        ('var', 'number'),
        [
          (
            'loop_assign',
            'integral',
            (
              'add',
              ('var', 'integral'),
              (
                'mul',
                (
                  'index',
                  'values',
                  ('imaginary'),
                ),
                ('var', 'grid_spacing'),
              ),
            ),
          ),
        ],
      )
    body:
      ('var', 'integral')
  dot_product:
    params: [('first', ('tensor', [('m', 'invariant')])), ('second', ('tensor', [('n', 'invariant')])), ('number', 'ℕ')]
    statements:
      (
        'body_decl',
        'value',
        'ℝ',
        ('num', 0.0),
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        ('var', 'number'),
        [
          (
            'loop_assign',
            'value',
            (
              'add',
              ('var', 'value'),
              (
                'mul',
                (
                  'index',
                  'first',
                  ('imaginary'),
                ),
                (
                  'index',
                  'second',
                  ('imaginary'),
                ),
              ),
            ),
          ),
        ],
      )
    body:
      ('var', 'value')
  normalize_vector:
    params: [('values', ('tensor', [('m', 'invariant')])), ('number', 'ℕ')]
    statements:
      (
        'body_decl',
        'result',
        (
          'tensor',
          [
            ('number', 'invariant'),
          ],
        ),
        (
          'call',
          'zero_array',
          [
            ('var', 'number'),
          ],
        ),
      )
      (
        'body_decl',
        'norm',
        'ℝ',
        (
          'call',
          'sqrt',
          [
            (
              'call',
              'dot_product',
              [
                ('var', 'values'),
                ('var', 'values'),
                ('var', 'number'),
              ],
            ),
          ],
        ),
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        ('var', 'number'),
        [
          (
            'loop_index_assign_nd',
            'result',
            [
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'div',
              (
                'index',
                'values',
                ('imaginary'),
              ),
              ('var', 'norm'),
            ),
          ),
        ],
      )
    body:
      ('var', 'result')
  apply_hamiltonian:
    params: [('wavefunction', ('tensor', [('m', 'invariant')])), ('potential', ('tensor', [('n', 'invariant')])), ('kinetic_coefficient', 'ℝ'), ('number', 'ℕ')]
    statements:
      (
        'body_decl',
        'result',
        (
          'tensor',
          [
            ('number', 'invariant'),
          ],
        ),
        (
          'call',
          'zero_array',
          [
            ('var', 'number'),
          ],
        ),
      )
      (
        'body_index_assign',
        'result',
        ('num', 0),
        (
          'sub',
          (
            'mul',
            (
              'add',
              (
                'mul',
                ('num', 2.0),
                ('var', 'kinetic_coefficient'),
              ),
              (
                'index',
                'potential',
                ('num', 0),
              ),
            ),
            (
              'index',
              'wavefunction',
              ('num', 0),
            ),
          ),
          (
            'mul',
            ('var', 'kinetic_coefficient'),
            (
              'index',
              'wavefunction',
              ('num', 1),
            ),
          ),
        ),
      )
      (
        'body_for_range',
        'i',
        ('num', 1),
        (
          'sub',
          ('var', 'number'),
          ('num', 1),
        ),
        [
          (
            'loop_index_assign_nd',
            'result',
            [
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'sub',
              (
                'add',
                (
                  'mul',
                  (
                    'neg',
                    ('var', 'kinetic_coefficient'),
                  ),
                  (
                    'index',
                    'wavefunction',
                    (
                      'sub',
                      ('imaginary'),
                      ('num', 1),
                    ),
                  ),
                ),
                (
                  'mul',
                  (
                    'add',
                    (
                      'mul',
                      ('num', 2.0),
                      ('var', 'kinetic_coefficient'),
                    ),
                    (
                      'index',
                      'potential',
                      ('imaginary'),
                    ),
                  ),
                  (
                    'index',
                    'wavefunction',
                    ('imaginary'),
                  ),
                ),
              ),
              (
                'mul',
                ('var', 'kinetic_coefficient'),
                (
                  'index',
                  'wavefunction',
                  (
                    'add',
                    ('imaginary'),
                    ('num', 1),
                  ),
                ),
              ),
            ),
          ),
        ],
      )
      (
        'body_index_assign',
        'result',
        (
          'sub',
          ('var', 'number'),
          ('num', 1),
        ),
        (
          'add',
          (
            'mul',
            (
              'neg',
              ('var', 'kinetic_coefficient'),
            ),
            (
              'index',
              'wavefunction',
              (
                'sub',
                ('var', 'number'),
                ('num', 2),
              ),
            ),
          ),
          (
            'mul',
            (
              'add',
              (
                'mul',
                ('num', 2.0),
                ('var', 'kinetic_coefficient'),
              ),
              (
                'index',
                'potential',
                (
                  'sub',
                  ('var', 'number'),
                  ('num', 1),
                ),
              ),
            ),
            (
              'index',
              'wavefunction',
              (
                'sub',
                ('var', 'number'),
                ('num', 1),
              ),
            ),
          ),
        ),
      )
    body:
      ('var', 'result')
  jacobi_diagonalize:
    params: []
    statements:
      (
        'body_decl',
        'jacobi_not_converged',
        'ℕ',
        ('num', 0),
      )
      (
        'body_decl',
        'jacobi_converged',
        'ℕ',
        ('num', 1),
      )
      (
        'body_decl',
        'jacobi_converged_local',
        'ℕ',
        ('var', 'jacobi_not_converged'),
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        ('var', 'Krylov_dimension'),
        [
          (
            'loop_index_assign_nd',
            'Ritz_vectors',
            [
              (
                'index_item',
                ('imaginary'),
              ),
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            ('num', 1.0),
          ),
          (
            'loop_for_range',
            'j',
            ('num', 0),
            ('var', 'Krylov_dimension'),
            [
              (
                'loop_index_assign_nd',
                'projected_work_matrix',
                [
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                  (
                    'index_item',
                    ('var', 'j'),
                  ),
                ],
                (
                  'indexN',
                  'projected_hamiltonian',
                  [
                    (
                      'index_item',
                      ('imaginary'),
                    ),
                    (
                      'index_item',
                      ('var', 'j'),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ],
      )
      (
        'body_for_range',
        'rotation',
        ('num', 0),
        ('var', 'jacobi_maximum_rotations'),
        [
          (
            'loop_assign',
            'jacobi_p',
            ('num', 0),
          ),
          (
            'loop_assign',
            'jacobi_q',
            ('num', 1),
          ),
          (
            'loop_assign',
            'jacobi_largest',
            (
              'call',
              'abs',
              [
                (
                  'indexN',
                  'projected_work_matrix',
                  [
                    (
                      'index_item',
                      ('num', 0),
                    ),
                    (
                      'index_item',
                      ('num', 1),
                    ),
                  ],
                ),
              ],
            ),
          ),
          (
            'loop_for_range',
            'i',
            ('num', 0),
            ('var', 'Krylov_dimension'),
            [
              (
                'loop_for_range',
                'j',
                (
                  'add',
                  ('imaginary'),
                  ('num', 1),
                ),
                ('var', 'Krylov_dimension'),
                [
                  (
                    'loop_if',
                    (
                      'cond_gt',
                      (
                        'call',
                        'abs',
                        [
                          (
                            'indexN',
                            'projected_work_matrix',
                            [
                              (
                                'index_item',
                                ('imaginary'),
                              ),
                              (
                                'index_item',
                                ('var', 'j'),
                              ),
                            ],
                          ),
                        ],
                      ),
                      ('var', 'jacobi_largest'),
                    ),
                    [
                      (
                        'loop_assign',
                        'jacobi_largest',
                        (
                          'call',
                          'abs',
                          [
                            (
                              'indexN',
                              'projected_work_matrix',
                              [
                                (
                                  'index_item',
                                  ('imaginary'),
                                ),
                                (
                                  'index_item',
                                  ('var', 'j'),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      (
                        'loop_assign',
                        'jacobi_p',
                        ('imaginary'),
                      ),
                      (
                        'loop_assign',
                        'jacobi_q',
                        ('var', 'j'),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
          (
            'loop_if',
            (
              'cond_eq',
              ('var', 'jacobi_converged_local'),
              ('var', 'jacobi_not_converged'),
            ),
            [
              (
                'loop_if',
                (
                  'cond_leq',
                  ('var', 'jacobi_largest'),
                  ('var', 'jacobi_tolerance'),
                ),
                [
                  (
                    'loop_assign',
                    'jacobi_converged_local',
                    ('var', 'jacobi_converged'),
                  ),
                ],
              ),
            ],
          ),
          (
            'loop_if',
            (
              'cond_gt',
              ('var', 'jacobi_largest'),
              ('var', 'jacobi_tolerance'),
            ),
            [
              (
                'loop_assign',
                'jacobi_angle',
                (
                  'mul',
                  ('num', 0.5),
                  (
                    'call',
                    'atan2',
                    [
                      (
                        'mul',
                        ('num', 2.0),
                        (
                          'indexN',
                          'projected_work_matrix',
                          [
                            (
                              'index_item',
                              ('var', 'jacobi_p'),
                            ),
                            (
                              'index_item',
                              ('var', 'jacobi_q'),
                            ),
                          ],
                        ),
                      ),
                      (
                        'sub',
                        (
                          'indexN',
                          'projected_work_matrix',
                          [
                            (
                              'index_item',
                              ('var', 'jacobi_q'),
                            ),
                            (
                              'index_item',
                              ('var', 'jacobi_q'),
                            ),
                          ],
                        ),
                        (
                          'indexN',
                          'projected_work_matrix',
                          [
                            (
                              'index_item',
                              ('var', 'jacobi_p'),
                            ),
                            (
                              'index_item',
                              ('var', 'jacobi_p'),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              (
                'loop_assign',
                'jacobi_cosine',
                (
                  'call',
                  'cos',
                  [
                    ('var', 'jacobi_angle'),
                  ],
                ),
              ),
              (
                'loop_assign',
                'jacobi_sine',
                (
                  'call',
                  'sin',
                  [
                    ('var', 'jacobi_angle'),
                  ],
                ),
              ),
              (
                'loop_assign',
                'jacobi_app',
                (
                  'add',
                  (
                    'indexN',
                    'projected_work_matrix',
                    [
                      (
                        'index_item',
                        ('var', 'jacobi_p'),
                      ),
                      (
                        'index_item',
                        ('var', 'jacobi_p'),
                      ),
                    ],
                  ),
                  ('num', 0.0),
                ),
              ),
              (
                'loop_assign',
                'jacobi_aqq',
                (
                  'add',
                  (
                    'indexN',
                    'projected_work_matrix',
                    [
                      (
                        'index_item',
                        ('var', 'jacobi_q'),
                      ),
                      (
                        'index_item',
                        ('var', 'jacobi_q'),
                      ),
                    ],
                  ),
                  ('num', 0.0),
                ),
              ),
              (
                'loop_assign',
                'jacobi_apq',
                (
                  'add',
                  (
                    'indexN',
                    'projected_work_matrix',
                    [
                      (
                        'index_item',
                        ('var', 'jacobi_p'),
                      ),
                      (
                        'index_item',
                        ('var', 'jacobi_q'),
                      ),
                    ],
                  ),
                  ('num', 0.0),
                ),
              ),
              (
                'loop_for_range',
                'k',
                ('num', 0),
                ('var', 'Krylov_dimension'),
                [
                  (
                    'loop_if',
                    (
                      'cond_neq',
                      ('var', 'k'),
                      ('var', 'jacobi_p'),
                    ),
                    [
                      (
                        'loop_if',
                        (
                          'cond_neq',
                          ('var', 'k'),
                          ('var', 'jacobi_q'),
                        ),
                        [
                          (
                            'loop_assign',
                            'jacobi_akp',
                            (
                              'add',
                              (
                                'indexN',
                                'projected_work_matrix',
                                [
                                  (
                                    'index_item',
                                    ('var', 'k'),
                                  ),
                                  (
                                    'index_item',
                                    ('var', 'jacobi_p'),
                                  ),
                                ],
                              ),
                              ('num', 0.0),
                            ),
                          ),
                          (
                            'loop_assign',
                            'jacobi_akq',
                            (
                              'add',
                              (
                                'indexN',
                                'projected_work_matrix',
                                [
                                  (
                                    'index_item',
                                    ('var', 'k'),
                                  ),
                                  (
                                    'index_item',
                                    ('var', 'jacobi_q'),
                                  ),
                                ],
                              ),
                              ('num', 0.0),
                            ),
                          ),
                          (
                            'loop_index_assign_nd',
                            'projected_work_matrix',
                            [
                              (
                                'index_item',
                                ('var', 'k'),
                              ),
                              (
                                'index_item',
                                ('var', 'jacobi_p'),
                              ),
                            ],
                            (
                              'sub',
                              (
                                'mul',
                                ('var', 'jacobi_cosine'),
                                ('var', 'jacobi_akp'),
                              ),
                              (
                                'mul',
                                ('var', 'jacobi_sine'),
                                ('var', 'jacobi_akq'),
                              ),
                            ),
                          ),
                          (
                            'loop_index_assign_nd',
                            'projected_work_matrix',
                            [
                              (
                                'index_item',
                                ('var', 'jacobi_p'),
                              ),
                              (
                                'index_item',
                                ('var', 'k'),
                              ),
                            ],
                            (
                              'indexN',
                              'projected_work_matrix',
                              [
                                (
                                  'index_item',
                                  ('var', 'k'),
                                ),
                                (
                                  'index_item',
                                  ('var', 'jacobi_p'),
                                ),
                              ],
                            ),
                          ),
                          (
                            'loop_index_assign_nd',
                            'projected_work_matrix',
                            [
                              (
                                'index_item',
                                ('var', 'k'),
                              ),
                              (
                                'index_item',
                                ('var', 'jacobi_q'),
                              ),
                            ],
                            (
                              'add',
                              (
                                'mul',
                                ('var', 'jacobi_sine'),
                                ('var', 'jacobi_akp'),
                              ),
                              (
                                'mul',
                                ('var', 'jacobi_cosine'),
                                ('var', 'jacobi_akq'),
                              ),
                            ),
                          ),
                          (
                            'loop_index_assign_nd',
                            'projected_work_matrix',
                            [
                              (
                                'index_item',
                                ('var', 'jacobi_q'),
                              ),
                              (
                                'index_item',
                                ('var', 'k'),
                              ),
                            ],
                            (
                              'indexN',
                              'projected_work_matrix',
                              [
                                (
                                  'index_item',
                                  ('var', 'k'),
                                ),
                                (
                                  'index_item',
                                  ('var', 'jacobi_q'),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
              (
                'loop_index_assign_nd',
                'projected_work_matrix',
                [
                  (
                    'index_item',
                    ('var', 'jacobi_p'),
                  ),
                  (
                    'index_item',
                    ('var', 'jacobi_p'),
                  ),
                ],
                (
                  'add',
                  (
                    'sub',
                    (
                      'mul',
                      (
                        'mul',
                        ('var', 'jacobi_cosine'),
                        ('var', 'jacobi_cosine'),
                      ),
                      ('var', 'jacobi_app'),
                    ),
                    (
                      'mul',
                      (
                        'mul',
                        (
                          'mul',
                          ('num', 2.0),
                          ('var', 'jacobi_sine'),
                        ),
                        ('var', 'jacobi_cosine'),
                      ),
                      ('var', 'jacobi_apq'),
                    ),
                  ),
                  (
                    'mul',
                    (
                      'mul',
                      ('var', 'jacobi_sine'),
                      ('var', 'jacobi_sine'),
                    ),
                    ('var', 'jacobi_aqq'),
                  ),
                ),
              ),
              (
                'loop_index_assign_nd',
                'projected_work_matrix',
                [
                  (
                    'index_item',
                    ('var', 'jacobi_q'),
                  ),
                  (
                    'index_item',
                    ('var', 'jacobi_q'),
                  ),
                ],
                (
                  'add',
                  (
                    'add',
                    (
                      'mul',
                      (
                        'mul',
                        ('var', 'jacobi_sine'),
                        ('var', 'jacobi_sine'),
                      ),
                      ('var', 'jacobi_app'),
                    ),
                    (
                      'mul',
                      (
                        'mul',
                        (
                          'mul',
                          ('num', 2.0),
                          ('var', 'jacobi_sine'),
                        ),
                        ('var', 'jacobi_cosine'),
                      ),
                      ('var', 'jacobi_apq'),
                    ),
                  ),
                  (
                    'mul',
                    (
                      'mul',
                      ('var', 'jacobi_cosine'),
                      ('var', 'jacobi_cosine'),
                    ),
                    ('var', 'jacobi_aqq'),
                  ),
                ),
              ),
              (
                'loop_index_assign_nd',
                'projected_work_matrix',
                [
                  (
                    'index_item',
                    ('var', 'jacobi_p'),
                  ),
                  (
                    'index_item',
                    ('var', 'jacobi_q'),
                  ),
                ],
                ('num', 0.0),
              ),
              (
                'loop_index_assign_nd',
                'projected_work_matrix',
                [
                  (
                    'index_item',
                    ('var', 'jacobi_q'),
                  ),
                  (
                    'index_item',
                    ('var', 'jacobi_p'),
                  ),
                ],
                ('num', 0.0),
              ),
              (
                'loop_for_range',
                'k',
                ('num', 0),
                ('var', 'Krylov_dimension'),
                [
                  (
                    'loop_assign',
                    'jacobi_vkp',
                    (
                      'add',
                      (
                        'indexN',
                        'Ritz_vectors',
                        [
                          (
                            'index_item',
                            ('var', 'k'),
                          ),
                          (
                            'index_item',
                            ('var', 'jacobi_p'),
                          ),
                        ],
                      ),
                      ('num', 0.0),
                    ),
                  ),
                  (
                    'loop_assign',
                    'jacobi_vkq',
                    (
                      'add',
                      (
                        'indexN',
                        'Ritz_vectors',
                        [
                          (
                            'index_item',
                            ('var', 'k'),
                          ),
                          (
                            'index_item',
                            ('var', 'jacobi_q'),
                          ),
                        ],
                      ),
                      ('num', 0.0),
                    ),
                  ),
                  (
                    'loop_index_assign_nd',
                    'Ritz_vectors',
                    [
                      (
                        'index_item',
                        ('var', 'k'),
                      ),
                      (
                        'index_item',
                        ('var', 'jacobi_p'),
                      ),
                    ],
                    (
                      'sub',
                      (
                        'mul',
                        ('var', 'jacobi_cosine'),
                        ('var', 'jacobi_vkp'),
                      ),
                      (
                        'mul',
                        ('var', 'jacobi_sine'),
                        ('var', 'jacobi_vkq'),
                      ),
                    ),
                  ),
                  (
                    'loop_index_assign_nd',
                    'Ritz_vectors',
                    [
                      (
                        'index_item',
                        ('var', 'k'),
                      ),
                      (
                        'index_item',
                        ('var', 'jacobi_q'),
                      ),
                    ],
                    (
                      'add',
                      (
                        'mul',
                        ('var', 'jacobi_sine'),
                        ('var', 'jacobi_vkp'),
                      ),
                      (
                        'mul',
                        ('var', 'jacobi_cosine'),
                        ('var', 'jacobi_vkq'),
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ],
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        ('var', 'Krylov_dimension'),
        [
          (
            'loop_index_assign_nd',
            'Ritz_values',
            [
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'indexN',
              'projected_work_matrix',
              [
                (
                  'index_item',
                  ('imaginary'),
                ),
                (
                  'index_item',
                  ('imaginary'),
                ),
              ],
            ),
          ),
        ],
      )
      (
        'body_for_range',
        'i',
        ('num', 0),
        (
          'sub',
          ('var', 'Krylov_dimension'),
          ('num', 1),
        ),
        [
          (
            'loop_assign',
            'jacobi_minimum',
            ('imaginary'),
          ),
          (
            'loop_for_range',
            'j',
            (
              'add',
              ('imaginary'),
              ('num', 1),
            ),
            ('var', 'Krylov_dimension'),
            [
              (
                'loop_if',
                (
                  'cond_lt',
                  (
                    'index',
                    'Ritz_values',
                    ('var', 'j'),
                  ),
                  (
                    'index',
                    'Ritz_values',
                    ('var', 'jacobi_minimum'),
                  ),
                ),
                [
                  (
                    'loop_assign',
                    'jacobi_minimum',
                    ('var', 'j'),
                  ),
                ],
              ),
            ],
          ),
          (
            'loop_if',
            (
              'cond_neq',
              ('var', 'jacobi_minimum'),
              ('imaginary'),
            ),
            [
              (
                'loop_assign',
                'jacobi_temporary_value',
                (
                  'add',
                  (
                    'index',
                    'Ritz_values',
                    ('imaginary'),
                  ),
                  ('num', 0.0),
                ),
              ),
              (
                'loop_index_assign_nd',
                'Ritz_values',
                [
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
                (
                  'index',
                  'Ritz_values',
                  ('var', 'jacobi_minimum'),
                ),
              ),
              (
                'loop_index_assign_nd',
                'Ritz_values',
                [
                  (
                    'index_item',
                    ('var', 'jacobi_minimum'),
                  ),
                ],
                ('var', 'jacobi_temporary_value'),
              ),
              (
                'loop_for_range',
                'k',
                ('num', 0),
                ('var', 'Krylov_dimension'),
                [
                  (
                    'loop_assign',
                    'jacobi_temporary_vector',
                    (
                      'add',
                      (
                        'indexN',
                        'Ritz_vectors',
                        [
                          (
                            'index_item',
                            ('var', 'k'),
                          ),
                          (
                            'index_item',
                            ('imaginary'),
                          ),
                        ],
                      ),
                      ('num', 0.0),
                    ),
                  ),
                  (
                    'loop_index_assign_nd',
                    'Ritz_vectors',
                    [
                      (
                        'index_item',
                        ('var', 'k'),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                    (
                      'indexN',
                      'Ritz_vectors',
                      [
                        (
                          'index_item',
                          ('var', 'k'),
                        ),
                        (
                          'index_item',
                          ('var', 'jacobi_minimum'),
                        ),
                      ],
                    ),
                  ),
                  (
                    'loop_index_assign_nd',
                    'Ritz_vectors',
                    [
                      (
                        'index_item',
                        ('var', 'k'),
                      ),
                      (
                        'index_item',
                        ('var', 'jacobi_minimum'),
                      ),
                    ],
                    ('var', 'jacobi_temporary_vector'),
                  ),
                ],
              ),
            ],
          ),
        ],
      )
    body:
      ('var', 'jacobi_converged_local')

Classes:

Program:
  ('func_def', 'zero_array')
  ('func_def', 'zero_matrix')
  ('func_def', 'linspace')
  ('func_def', 'integrate')
  ('func_def', 'dot_product')
  ('func_def', 'normalize_vector')
  ('func_def', 'apply_hamiltonian')
  ('func_def', 'jacobi_diagonalize')
  (
    'decl',
    'atomic_mass_unit',
    'ℝ',
    ('num', 1.6605390666e-27),
    108,
  )
  (
    'decl',
    'hbar',
    'ℝ',
    ('num', 1.054571817e-34),
    109,
  )
  (
    'decl',
    'planck_constant',
    'ℝ',
    ('num', 6.62607015e-34),
    110,
  )
  (
    'decl',
    'speed_of_light_cm',
    'ℝ',
    ('num', 29979245800.0),
    111,
  )
  (
    'decl',
    'electron_volt',
    'ℝ',
    ('num', 1.602176634e-19),
    112,
  )
  (
    'decl',
    'mass_H_u',
    'ℝ',
    ('num', 1.00784),
    113,
  )
  (
    'decl',
    'mass_Cl_u',
    'ℝ',
    ('num', 35.45),
    114,
  )
  (
    'decl',
    'mass_H',
    'ℝ',
    (
      'mul',
      ('var', 'mass_H_u'),
      ('var', 'atomic_mass_unit'),
    ),
    115,
  )
  (
    'decl',
    'mass_Cl',
    'ℝ',
    (
      'mul',
      ('var', 'mass_Cl_u'),
      ('var', 'atomic_mass_unit'),
    ),
    116,
  )
  (
    'decl',
    'reduced_mass_HCl',
    'ℝ',
    (
      'div',
      (
        'mul',
        ('var', 'mass_H'),
        ('var', 'mass_Cl'),
      ),
      (
        'add',
        ('var', 'mass_H'),
        ('var', 'mass_Cl'),
      ),
    ),
    117,
  )
  (
    'decl',
    'dissociation_energy_eV',
    'ℝ',
    ('num', 4.61907),
    118,
  )
  (
    'decl',
    'dissociation_energy_J',
    'ℝ',
    (
      'mul',
      ('var', 'dissociation_energy_eV'),
      ('var', 'electron_volt'),
    ),
    119,
  )
  (
    'decl',
    'equilibrium_distance',
    'ℝ',
    ('num', 1.2746e-10),
    120,
  )
  (
    'decl',
    'morse_a',
    'ℝ',
    ('num', 18680000000.0),
    121,
  )
  (
    'decl',
    'N_grid',
    'ℕ',
    ('num', 100),
    122,
  )
  (
    'decl',
    'N_levels',
    'ℕ',
    ('num', 2),
    123,
  )
  (
    'decl',
    'block_size',
    'ℕ',
    ('num', 2),
    124,
  )
  (
    'decl',
    'Krylov_dimension',
    'ℕ',
    ('num', 40),
    125,
  )
  (
    'decl',
    'r_min',
    'ℝ',
    ('num', 5e-11),
    126,
  )
  (
    'decl',
    'r_max',
    'ℝ',
    ('num', 2e-10),
    127,
  )
  (
    'decl',
    'grid_spacing',
    'ℝ',
    (
      'div',
      (
        'sub',
        ('var', 'r_max'),
        ('var', 'r_min'),
      ),
      (
        'add',
        ('var', 'N_grid'),
        ('num', 1),
      ),
    ),
    128,
  )
  (
    'decl',
    'r_start',
    'ℝ',
    (
      'add',
      ('var', 'r_min'),
      ('var', 'grid_spacing'),
    ),
    129,
  )
  (
    'decl',
    'r_end',
    'ℝ',
    (
      'sub',
      ('var', 'r_max'),
      ('var', 'grid_spacing'),
    ),
    130,
  )
  (
    'decl',
    'bond_distance',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'linspace',
      [
        ('var', 'r_start'),
        ('var', 'r_end'),
        ('var', 'N_grid'),
      ],
    ),
    131,
  )
  (
    'decl',
    'distance_from_equilibrium',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'sub',
      ('var', 'bond_distance'),
      ('var', 'equilibrium_distance'),
    ),
    132,
  )
  (
    'decl',
    'morse_exponential',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'exp',
      [
        (
          'mul',
          (
            'neg',
            ('var', 'morse_a'),
          ),
          ('var', 'distance_from_equilibrium'),
        ),
      ],
    ),
    133,
  )
  (
    'decl',
    'morse_difference',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'sub',
      ('num', 1.0),
      ('var', 'morse_exponential'),
    ),
    134,
  )
  (
    'decl',
    'potential_J',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'mul',
      (
        'mul',
        ('var', 'dissociation_energy_J'),
        ('var', 'morse_difference'),
      ),
      ('var', 'morse_difference'),
    ),
    135,
  )
  (
    'decl',
    'potential_eV',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'div',
      ('var', 'potential_J'),
      ('var', 'electron_volt'),
    ),
    136,
  )
  (
    'decl',
    'kinetic_coefficient_J',
    'ℝ',
    (
      'mul',
      (
        'div',
        ('var', 'hbar'),
        ('var', 'grid_spacing'),
      ),
      (
        'div',
        ('var', 'hbar'),
        (
          'mul',
          (
            'mul',
            ('num', 2.0),
            ('var', 'reduced_mass_HCl'),
          ),
          ('var', 'grid_spacing'),
        ),
      ),
    ),
    137,
  )
  (
    'decl',
    'kinetic_coefficient_eV',
    'ℝ',
    (
      'div',
      ('var', 'kinetic_coefficient_J'),
      ('var', 'electron_volt'),
    ),
    138,
  )
  (
    'decl',
    'trial_width',
    'ℝ',
    ('num', 2e-11),
    139,
  )
  (
    'decl',
    'gaussian_argument',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'div',
      ('var', 'distance_from_equilibrium'),
      ('var', 'trial_width'),
    ),
    140,
  )
  (
    'decl',
    'gaussian_envelope',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'exp',
      [
        (
          'mul',
          (
            'neg',
            ('var', 'gaussian_argument'),
          ),
          ('var', 'gaussian_argument'),
        ),
      ],
    ),
    141,
  )
  (
    'decl',
    'Krylov_basis',
    (
      'tensor',
      [
        ('Krylov_dimension', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'Krylov_dimension'),
        ('var', 'N_grid'),
      ],
    ),
    142,
  )
  (
    'decl',
    'H_Krylov',
    (
      'tensor',
      [
        ('Krylov_dimension', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'Krylov_dimension'),
        ('var', 'N_grid'),
      ],
    ),
    143,
  )
  (
    'decl',
    'projected_hamiltonian',
    (
      'tensor',
      [
        ('Krylov_dimension', 'invariant'),
        ('Krylov_dimension', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'Krylov_dimension'),
        ('var', 'Krylov_dimension'),
      ],
    ),
    144,
  )
  (
    'decl',
    'projected_work_matrix',
    (
      'tensor',
      [
        ('Krylov_dimension', 'invariant'),
        ('Krylov_dimension', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'Krylov_dimension'),
        ('var', 'Krylov_dimension'),
      ],
    ),
    145,
  )
  (
    'decl',
    'Ritz_vectors',
    (
      'tensor',
      [
        ('Krylov_dimension', 'invariant'),
        ('Krylov_dimension', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'Krylov_dimension'),
        ('var', 'Krylov_dimension'),
      ],
    ),
    146,
  )
  (
    'decl',
    'Ritz_values',
    (
      'tensor',
      [
        ('Krylov_dimension', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_array',
      [
        ('var', 'Krylov_dimension'),
      ],
    ),
    147,
  )
  (
    'decl',
    'jacobi_tolerance',
    'ℝ',
    ('num', 1e-07),
    148,
  )
  (
    'decl',
    'jacobi_maximum_rotations',
    'ℕ',
    ('num', 10000),
    149,
  )
  (
    'decl',
    'jacobi_largest',
    'ℝ',
    ('num', 0.0),
    150,
  )
  (
    'decl',
    'jacobi_angle',
    'ℝ',
    ('num', 0.0),
    151,
  )
  (
    'decl',
    'jacobi_cosine',
    'ℝ',
    ('num', 0.0),
    152,
  )
  (
    'decl',
    'jacobi_sine',
    'ℝ',
    ('num', 0.0),
    153,
  )
  (
    'decl',
    'jacobi_app',
    'ℝ',
    ('num', 0.0),
    154,
  )
  (
    'decl',
    'jacobi_aqq',
    'ℝ',
    ('num', 0.0),
    155,
  )
  (
    'decl',
    'jacobi_apq',
    'ℝ',
    ('num', 0.0),
    156,
  )
  (
    'decl',
    'jacobi_akp',
    'ℝ',
    ('num', 0.0),
    157,
  )
  (
    'decl',
    'jacobi_akq',
    'ℝ',
    ('num', 0.0),
    158,
  )
  (
    'decl',
    'jacobi_vkp',
    'ℝ',
    ('num', 0.0),
    159,
  )
  (
    'decl',
    'jacobi_vkq',
    'ℝ',
    ('num', 0.0),
    160,
  )
  (
    'decl',
    'jacobi_temporary_value',
    'ℝ',
    ('num', 0.0),
    161,
  )
  (
    'decl',
    'jacobi_temporary_vector',
    'ℝ',
    ('num', 0.0),
    162,
  )
  (
    'decl',
    'jacobi_p',
    'ℕ',
    ('num', 0),
    163,
  )
  (
    'decl',
    'jacobi_q',
    'ℕ',
    ('num', 1),
    164,
  )
  (
    'decl',
    'jacobi_minimum',
    'ℕ',
    ('num', 0),
    165,
  )
  (
    'decl',
    'candidate',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_array',
      [
        ('var', 'N_grid'),
      ],
    ),
    166,
  )
  (
    'decl',
    'overlap',
    'ℝ',
    ('num', 0.0),
    167,
  )
  (
    'decl',
    'candidate_norm',
    'ℝ',
    ('num', 0.0),
    168,
  )
  (
    'for_loop_range',
    'i',
    ('num', 0),
    ('var', 'N_grid'),
    [
      (
        'for_index_assign_nd',
        'Krylov_basis',
        [
          (
            'index_item',
            ('num', 0),
          ),
          (
            'index_item',
            ('imaginary'),
          ),
        ],
        (
          'index',
          'gaussian_envelope',
          ('imaginary'),
        ),
      ),
    ],
    170,
  )
  (
    'index_assign',
    'Krylov_basis',
    0,
    (
      'call',
      'normalize_vector',
      [
        (
          'index',
          'Krylov_basis',
          ('num', 0),
        ),
        ('var', 'N_grid'),
      ],
    ),
    172,
  )
  (
    'for_loop_range',
    'q_index',
    ('num', 1),
    ('var', 'Krylov_dimension'),
    [
      (
        'for_if_else',
        (
          'cond_lt',
          ('var', 'q_index'),
          ('var', 'block_size'),
        ),
        [
          (
            'for_loop_range',
            'i',
            ('num', 0),
            ('var', 'N_grid'),
            [
              (
                'for_index_assign_nd',
                'candidate',
                [
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
                (
                  'mul',
                  (
                    'index',
                    'gaussian_argument',
                    ('imaginary'),
                  ),
                  (
                    'indexN',
                    'Krylov_basis',
                    [
                      (
                        'index_item',
                        (
                          'sub',
                          ('var', 'q_index'),
                          ('num', 1),
                        ),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                  ),
                ),
              ),
            ],
            176,
          ),
        ],
        [
          (
            'for_assign',
            'candidate',
            (
              'call',
              'apply_hamiltonian',
              [
                (
                  'index',
                  'Krylov_basis',
                  (
                    'sub',
                    ('var', 'q_index'),
                    ('var', 'block_size'),
                  ),
                ),
                ('var', 'potential_eV'),
                ('var', 'kinetic_coefficient_eV'),
                ('var', 'N_grid'),
              ],
            ),
          ),
        ],
      ),
      (
        'for_loop_range',
        'orthogonalization_pass',
        ('num', 0),
        ('num', 2),
        [
          (
            'for_loop_range',
            'lower',
            ('num', 0),
            ('var', 'q_index'),
            [
              (
                'for_assign',
                'overlap',
                (
                  'call',
                  'dot_product',
                  [
                    (
                      'index',
                      'Krylov_basis',
                      ('var', 'lower'),
                    ),
                    ('var', 'candidate'),
                    ('var', 'N_grid'),
                  ],
                ),
              ),
              (
                'for_loop_range',
                'i',
                ('num', 0),
                ('var', 'N_grid'),
                [
                  (
                    'for_index_assign_nd',
                    'candidate',
                    [
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                    (
                      'sub',
                      (
                        'index',
                        'candidate',
                        ('imaginary'),
                      ),
                      (
                        'mul',
                        ('var', 'overlap'),
                        (
                          'indexN',
                          'Krylov_basis',
                          [
                            (
                              'index_item',
                              ('var', 'lower'),
                            ),
                            (
                              'index_item',
                              ('imaginary'),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
                183,
              ),
            ],
            181,
          ),
        ],
        180,
      ),
      (
        'for_assign',
        'candidate_norm',
        (
          'call',
          'sqrt',
          [
            (
              'call',
              'dot_product',
              [
                ('var', 'candidate'),
                ('var', 'candidate'),
                ('var', 'N_grid'),
              ],
            ),
          ],
        ),
      ),
      (
        'for_loop_range',
        'i',
        ('num', 0),
        ('var', 'N_grid'),
        [
          (
            'for_index_assign_nd',
            'Krylov_basis',
            [
              (
                'index_item',
                ('var', 'q_index'),
              ),
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'div',
              (
                'index',
                'candidate',
                ('imaginary'),
              ),
              ('var', 'candidate_norm'),
            ),
          ),
        ],
        186,
      ),
    ],
    174,
  )
  (
    'for_loop_range',
    'q_index',
    ('num', 0),
    ('var', 'Krylov_dimension'),
    [
      (
        'for_index_assign_nd',
        'H_Krylov',
        [
          (
            'index_item',
            ('var', 'q_index'),
          ),
        ],
        (
          'call',
          'apply_hamiltonian',
          [
            (
              'index',
              'Krylov_basis',
              ('var', 'q_index'),
            ),
            ('var', 'potential_eV'),
            ('var', 'kinetic_coefficient_eV'),
            ('var', 'N_grid'),
          ],
        ),
      ),
    ],
    188,
  )
  (
    'for_loop_range',
    'row',
    ('num', 0),
    ('var', 'Krylov_dimension'),
    [
      (
        'for_loop_range',
        'column',
        ('var', 'row'),
        ('var', 'Krylov_dimension'),
        [
          (
            'for_index_assign_nd',
            'projected_hamiltonian',
            [
              (
                'index_item',
                ('var', 'row'),
              ),
              (
                'index_item',
                ('var', 'column'),
              ),
            ],
            (
              'call',
              'dot_product',
              [
                (
                  'index',
                  'Krylov_basis',
                  ('var', 'row'),
                ),
                (
                  'index',
                  'H_Krylov',
                  ('var', 'column'),
                ),
                ('var', 'N_grid'),
              ],
            ),
          ),
          (
            'for_index_assign_nd',
            'projected_hamiltonian',
            [
              (
                'index_item',
                ('var', 'column'),
              ),
              (
                'index_item',
                ('var', 'row'),
              ),
            ],
            (
              'indexN',
              'projected_hamiltonian',
              [
                (
                  'index_item',
                  ('var', 'row'),
                ),
                (
                  'index_item',
                  ('var', 'column'),
                ),
              ],
            ),
          ),
        ],
        192,
      ),
    ],
    191,
  )
  (
    'expr',
    (
      'call',
      'jacobi_diagonalize',
      [],
    ),
    0,
  )
  (
    'decl',
    'vibrational_energies_eV',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_array',
      [
        ('var', 'N_levels'),
      ],
    ),
    197,
  )
  (
    'decl',
    'psi_raw',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'N_levels'),
        ('var', 'N_grid'),
      ],
    ),
    198,
  )
  (
    'for_loop_range',
    'n',
    ('num', 0),
    ('var', 'N_levels'),
    [
      (
        'for_index_assign_nd',
        'vibrational_energies_eV',
        [
          (
            'index_item',
            ('var', 'n'),
          ),
        ],
        (
          'index',
          'Ritz_values',
          ('var', 'n'),
        ),
      ),
      (
        'for_loop_range',
        'i',
        ('num', 0),
        ('var', 'N_grid'),
        [
          (
            'for_loop_range',
            'q_index',
            ('num', 0),
            ('var', 'Krylov_dimension'),
            [
              (
                'for_index_assign_nd',
                'psi_raw',
                [
                  (
                    'index_item',
                    ('var', 'n'),
                  ),
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
                (
                  'add',
                  (
                    'indexN',
                    'psi_raw',
                    [
                      (
                        'index_item',
                        ('var', 'n'),
                      ),
                      (
                        'index_item',
                        ('imaginary'),
                      ),
                    ],
                  ),
                  (
                    'mul',
                    (
                      'indexN',
                      'Krylov_basis',
                      [
                        (
                          'index_item',
                          ('var', 'q_index'),
                        ),
                        (
                          'index_item',
                          ('imaginary'),
                        ),
                      ],
                    ),
                    (
                      'indexN',
                      'Ritz_vectors',
                      [
                        (
                          'index_item',
                          ('var', 'q_index'),
                        ),
                        (
                          'index_item',
                          ('var', 'n'),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
            203,
          ),
        ],
        202,
      ),
      (
        'for_index_assign_nd',
        'psi_raw',
        [
          (
            'index_item',
            ('var', 'n'),
          ),
        ],
        (
          'call',
          'normalize_vector',
          [
            (
              'index',
              'psi_raw',
              ('var', 'n'),
            ),
            ('var', 'N_grid'),
          ],
        ),
      ),
    ],
    200,
  )
  (
    'decl',
    'normalization_factor',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_array',
      [
        ('var', 'N_levels'),
      ],
    ),
    207,
  )
  (
    'decl',
    'psi',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'call',
      'zero_matrix',
      [
        ('var', 'N_levels'),
        ('var', 'N_grid'),
      ],
    ),
    208,
  )
  (
    'for_loop_range',
    'n',
    ('num', 0),
    ('var', 'N_levels'),
    [
      (
        'for_index_assign_nd',
        'normalization_factor',
        [
          (
            'index_item',
            ('var', 'n'),
          ),
        ],
        (
          'call',
          'sqrt',
          [
            (
              'call',
              'integrate',
              [
                (
                  'mul',
                  (
                    'index',
                    'psi_raw',
                    ('var', 'n'),
                  ),
                  (
                    'index',
                    'psi_raw',
                    ('var', 'n'),
                  ),
                ),
                ('var', 'grid_spacing'),
                ('var', 'N_grid'),
              ],
            ),
          ],
        ),
      ),
      (
        'for_loop_range',
        'i',
        ('num', 0),
        ('var', 'N_grid'),
        [
          (
            'for_index_assign_nd',
            'psi',
            [
              (
                'index_item',
                ('var', 'n'),
              ),
              (
                'index_item',
                ('imaginary'),
              ),
            ],
            (
              'div',
              (
                'indexN',
                'psi_raw',
                [
                  (
                    'index_item',
                    ('var', 'n'),
                  ),
                  (
                    'index_item',
                    ('imaginary'),
                  ),
                ],
              ),
              (
                'index',
                'normalization_factor',
                ('var', 'n'),
              ),
            ),
          ),
        ],
        212,
      ),
    ],
    210,
  )
  (
    'decl',
    'transition_eV',
    'ℝ',
    (
      'sub',
      (
        'index',
        'vibrational_energies_eV',
        ('num', 1),
      ),
      (
        'index',
        'vibrational_energies_eV',
        ('num', 0),
      ),
    ),
    215,
  )
  (
    'decl',
    'transition_J',
    'ℝ',
    (
      'mul',
      ('var', 'transition_eV'),
      ('var', 'electron_volt'),
    ),
    216,
  )
  (
    'decl',
    'wavenumber',
    'ℝ',
    (
      'div',
      ('var', 'transition_J'),
      (
        'mul',
        ('var', 'planck_constant'),
        ('var', 'speed_of_light_cm'),
      ),
    ),
    217,
  )
  (
    'decl',
    'wavelength_micrometer',
    'ℝ',
    (
      'div',
      ('num', 10000.0),
      ('var', 'wavenumber'),
    ),
    218,
  )
  (
    'expr',
    (
      'call',
      'physika_print',
      [
        (
          'index',
          'vibrational_energies_eV',
          ('num', 0),
        ),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'call',
      'physika_print',
      [
        (
          'index',
          'vibrational_energies_eV',
          ('num', 1),
        ),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'call',
      'physika_print',
      [
        ('var', 'transition_eV'),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'call',
      'physika_print',
      [
        ('var', 'wavenumber'),
      ],
    ),
    0,
  )
  (
    'expr',
    (
      'call',
      'physika_print',
      [
        ('var', 'wavelength_micrometer'),
      ],
    ),
    0,
  )
  (
    'decl',
    'bond_distance_angstrom',
    (
      'tensor',
      [
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'div',
      ('var', 'bond_distance'),
      ('num', 1e-10),
    ),
    226,
  )
  (
    'decl',
    'equilibrium_distance_angstrom',
    'ℝ',
    (
      'div',
      ('var', 'equilibrium_distance'),
      ('num', 1e-10),
    ),
    227,
  )
  (
    'decl',
    'psi_angstrom',
    (
      'tensor',
      [
        ('N_levels', 'invariant'),
        ('N_grid', 'invariant'),
      ],
    ),
    (
      'mul',
      (
        'call',
        'sqrt',
        [
          ('num', 1e-10),
        ],
      ),
      ('var', 'psi'),
    ),
    228,
  )
